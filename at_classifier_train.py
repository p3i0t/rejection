import numpy as np

import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import resnet18, resnet34, resnet50
from utils import cal_parameters, get_dataset, AverageMeter


logger = logging.getLogger(__name__)

clip_min = 0.
clip_max = 1.


def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)


def train_epoch(classifier, data_loader, args, optimizer, scheduler):
    """
    Adversarial training one epoch with Fast FGSM.
    """
    classifier.train()

    # ajust according to std.
    eps = eval(args.epsilon)
    eps_iter = eval(args.epsilon_iter)

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        # start with uniform noise
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.requires_grad_()
        delta = clamp(delta, clip_min - x, clip_max - x)

        loss = F.cross_entropy(classifier(x + delta), y)
        grad_delta = torch.autograd.grad(loss, delta)[0].detach()  # get grad of noise

        # update delta with grad
        delta = (delta + torch.sign(grad_delta) * eps_iter).clamp_(-eps, eps)
        delta = clamp(delta, clip_min - x, clip_max - x)

        # real forward
        logits = classifier(x + delta)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


def attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts):
    """
    Perform PGD attack on one mini-batch.
    :param model: pytorch model.
    :param x: x of minibatch.
    :param y: y of minibatch.
    :param eps: L-infinite norm budget.
    :param eps_iter: step size for each iteration.
    :param attack_iters: number of iterations.
    :param restarts:  number of restart times
    :return: best adversarial perturbations delta in all restarts
    """
    assert x.device == y.device
    max_loss = torch.zeros_like(y).float()
    max_delta = torch.zeros_like(x)

    for i in range(restarts):
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.data = clamp(delta, clip_min - x, clip_max - x)
        delta.requires_grad = True

        for _ in range(attack_iters):
            logits = model(x + delta)
            # index = torch.where(output.max(1)[1] == y)
            index = torch.where(logits.argmax(dim=1) == y)  # get the correct predictions, pgd performed only on them
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # select & update
            grad = delta.grad.detach()
            delta_update = (delta[index] + eps_iter * torch.sign(grad[index])).clamp_(-eps, eps)
            delta_update = clamp(delta_update, clip_min - x[index], clip_max - x[index])

            # write back
            delta.data[index] = delta_update
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(x + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def eval_epoch(model, data_loader, args, adversarial=False):
    """Self-implemented PGD evaluation."""
    eps = eval(args.epsilon)
    eps_iter = eval(args.pgd_epsilon_iter)
    attack_iters = 40
    restarts = 2

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    model.eval()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        if adversarial is True:
            delta = attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts)
        else:
            delta = 0.

        with torch.no_grad():
            logits = model(x + delta)
            loss = F.cross_entropy(logits, y)

            loss_meter.update(loss.item(), x.size(0))
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='configs/at_base_config.yaml')
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

    if args.inference is True:
        classifier.load_state_dict(torch.load('{}_at.pth'.format(args.classifier_name)))
        logger.info('Load classifier from checkpoint')
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            args.lr_max,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)

        lr_steps = args.epochs * len(train_loader)
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                                          step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                             gamma=0.1)
        else:
            raise Exception("scheduler not implemented.")

        optimal_loss = 1e5
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_epoch(classifier, train_loader, args, optimizer, scheduler)
            lr = scheduler.get_lr()[0]
            logger.info('Epoch {}, lr:{:.4f}, loss:{:.4f}, Acc:{:.4f}'.format(epoch, lr, loss, acc))

            if loss < optimal_loss:
                optimal_loss = loss

                torch.save(classifier.state_dict(), 'resnet18_at.pth')

    clean_loss, clean_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
    adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True)
    logger.info('Clean loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))
    logger.info('Adversarial loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))


if __name__ == '__main__':
    run()
