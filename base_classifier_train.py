import numpy as np

import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models import resnet18, resnet34, resnet50
from utils import cal_parameters, get_dataset, AverageMeter


logger = logging.getLogger(__name__)


def run_epoch(classifier, data_loader, args, optimizer=None, scheduler=None):
    """
    Run one epoch on clean dataset.
    """
    if optimizer:
        classifier.train()
    else:
        classifier.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        output = classifier(x)
        loss = F.cross_entropy(output, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler:  # adjust learning rate
            scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (output.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train(classifier, train_loader, test_loader, args):
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    best_train_loss = np.inf
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(classifier, train_loader, args, optimizer=optimizer, scheduler=scheduler)
        lr = scheduler.get_lr()[0]
        logger.info('Epoch: {}, lr: {:.4f}, training loss: {:.4f}, acc: {:.4f}.'.format(epoch, lr, train_loss, train_acc))

        test_loss, test_acc = run_epoch(classifier, test_loader, args)
        logger.info("Test loss: {:.4f}, acc: {:.4f}".format(test_loss, test_acc))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_name = 'resnet18_wd{}.pth'.format(args.weight_decay)
            state = classifier.state_dict()

            torch.save(state, save_name)
            logger.info("==> New optimal training loss & saving checkpoint ...")


@hydra.main(config_path='configs/base_config.yaml')
def run(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)

    n_classes = args.get(args.dataset).n_classes
    classifier = resnet18(n_classes=n_classes).to(args.device)
    logger.info('Base classifier resnet18: # parameters {}'.format(cal_parameters(classifier)))

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.n_batch_train, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.n_batch_test, shuffle=False)

    if args.inference:
        save_name = 'resnet18_wd{}.pth'.format(args.weight_decay)
        classifier.load_state_dict(torch.load(save_name, map_location=lambda storage, loc: storage))
        loss, acc = run_epoch(classifier, test_loader, args)
        logger.info('Inference, test loss: {:.4f}, Acc: {:.4f}'.format(loss, acc))
    else:
        train(classifier, train_loader, test_loader, args)


if __name__ == '__main__':
    run()

