import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import augmentations

from models import resnet18

from utils import AverageMeter, cal_parameters

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def aug(image, preprocess, args):
    """Perform AugMix augmentations and compute mixture.

    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if args.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):  # size of composed augmentations set
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
        for _ in range(depth):   # compose one augmentation with depth number of single aug operation.
          op = np.random.choice(aug_list)
          image_aug = op(image_aug, args.aug_severity)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, args, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.args = args

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.args), aug(x, self.preprocess, self.args))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def jason_shanon_loss(prob_list):
    from functools import reduce
    # Clamp mixture distribution to avoid exploding KL divergence
    p_mix = reduce(lambda a, b: a + b, prob_list) / len(prob_list)
    p_mix = p_mix.clamp(1e-7, 1.).log()

    return reduce(lambda a, b: a + b, [F.kl_div(p_mix, p, reduction='batchmean') for p in prob_list]) / len(prob_list)


def train_epoch(classifier, train_loader, args, optimizer, scheduler):
    """Train for one epoch."""
    classifier.train()
    loss_meter = AverageMeter('loss')
    ce_meter = AverageMeter('ce_loss')
    js_meter = AverageMeter('js_loss')
    acc_meter = AverageMeter('acc_loss')

    for i, (x, y) in enumerate(train_loader):
        x_all = torch.cat(x, 0).to(args.device)
        y = y.to(args.device)
        logits_all = classifier(x_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(
          logits_all, x[0].size(0))

        # Cross-entropy is only computed on clean images
        ce_loss = F.cross_entropy(logits_clean, y)

        p_clean = logits_clean.softmax(dim=1)
        p_aug1 = logits_aug1.softmax(dim=1)
        p_aug2 = logits_aug2.softmax(dim=1)

        js_loss = jason_shanon_loss([p_clean, p_aug1, p_aug2])

        loss = ce_loss + 12 * js_loss

        loss_meter.update(loss.item(), x[0].size(0))
        ce_meter.update(ce_loss.item(), x[0].size(0))
        js_meter.update(js_loss.item(), x[0].size(0))
        acc = (logits_clean.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss_meter.avg, ce_meter.avg, js_meter.avg, acc_meter.avg


def eval_epoch(model, data_loader, args):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    model.eval()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss_meter.update(loss.item(), x.size(0))
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


class CorruptionDataset(Dataset):
    # for cifar10 and cifar100
    def __init__(self, x, y, transform=None):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[item]

    def __len__(self):
        return self.x.shape[0]


def eval_c(classifier, base_path, args):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        # preprocess = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize([mean_] * 3, [std_] * 3)])
        preprocess = transforms.ToTensor()

        x = np.load(base_path + corruption + '.npy')
        y = np.load(base_path + 'labels.npy').astype(np.int64)
        dataset = CorruptionDataset(x, y, transform=preprocess)

        test_loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc = eval_epoch(classifier, test_loader, args)
        corruption_accs.append(test_acc)

    return np.mean(corruption_accs)


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='configs/c_base_config.yaml')
def run(args: DictConfig) -> None:
    # Load datasets
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])

    preprocess = transforms.ToTensor()
    test_transform = preprocess

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            data_dir, train=False, transform=test_transform, download=True)
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C/')
        args.n_classes = 10
    else:
        train_data = datasets.CIFAR100(
            data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            data_dir, train=False, transform=test_transform, download=True)

        base_c_path = os.path.join(data_dir, 'CIFAR-100-C/')
        args.n_classes = 100

    train_data = AugMixDataset(train_data, preprocess, args, args.no_jsd)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    n_classes = args.get(args.dataset).n_classes
    classifier = resnet18(n_classes=n_classes).to(args.device)
    logger.info('Model resnet18, # parameters: {}'.format(cal_parameters(classifier)))

    cudnn.benchmark = True

    if args.inference:
        classifier.load_state_dict(torch.load('resnet18_c.pth'))
        test_loss, test_acc = eval_epoch(classifier, test_loader, args)
        logger.info('Clean Test CE:{:.4f}, acc:{:.4f}'.format(test_loss, test_acc))
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)

        best_loss = 1e5
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.n_epochs * len(train_loader),
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))

        for epoch in range(args.epochs):
            loss, ce_loss, js_loss, acc = train_epoch(classifier, train_loader,  args, optimizer, scheduler)

            lr = scheduler.get_lr()[0]
            logger.info('Epoch {}, lr:{:.4f}, loss:{:.4f}, CE:{:.4f}, JS:{:.4f}, Acc:{:.4f}'
                        .format(epoch + 1, lr, loss, ce_loss, js_loss, acc))

            test_loss, test_acc = eval_epoch(classifier, test_loader, args)
            logger.info('Clean test CE:{:.4f}, acc:{:.4f}'.format(test_loss, test_acc))

            if loss < best_loss:
                best_loss = loss
                logging.info('===> New optimal, save checkpoint ...')
                torch.save(classifier.state_dict(), 'resnet18_c.pth')

    test_c_acc = eval_c(classifier, base_c_path, args)
    logger.info('Mean Corruption Error:{:.4f}'.format(test_c_acc))


if __name__ == '__main__':
    run()

