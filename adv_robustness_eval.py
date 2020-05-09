import argparse
import sys
import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from advertorch.attacks import LinfPGDAttack
from models import resnet18, resnet34, resnet50
from sdim import SDIM
from utils import get_dataset, AverageMeter


logger = logging.getLogger(__name__)

suffix_dict = {'normal': '', 'at': '_at'}

clip_min = 0.
clip_max = 1.


@hydra.main(config_path='configs/adv_config.yaml')
def run(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)

    n_classes = args.get(args.dataset).n_classes
    rep_size = args.get(args.dataset).rep_size
    margin = args.get(args.dataset).margin
    
    classifier = resnet18(n_classes=n_classes).to(args.device)

    sdim = SDIM(disc_classifier=classifier,
                n_classes=n_classes,
                rep_size=rep_size,
                mi_units=args.mi_units,
                margin=margin).to(args.device)

    base_dir = hydra.utils.to_absolute_path('logs/sdim/{}'.format(args.dataset))
    save_name = 'SDIM_resnet18{}.pth'.format(suffix_dict[args.base_type])
    sdim.load_state_dict(torch.load(os.path.join(base_dir, save_name), map_location=lambda storage, loc: storage))

    if args.sample_likelihood:
        sample_cases(sdim, args)
    else:
        pgd_attack(sdim, args)


def sample_cases(sdim, args):
    sdim.eval()
    n_classes = args.get(args.dataset).n_classes

    sample_likelihood_dict = {}
    # logger.info('==> Corruption type: {}, severity level {}'.format(corruption_type, level))
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    x, y = next(iter(test_loader))
    x, y = x.to(args.device), y.long().to(args.device)

    def f_forward(x_, y_, image_name):
        with torch.no_grad():
            log_lik = sdim(x_)
        save_name = '{}.png'.format(image_name)
        save_image(x_, save_name, normalize=True)
        return log_lik[:, y_].item()

    sample_likelihood_dict['original'] = f_forward(x, y, 'original')

    eps_2 = 2 / 255
    eps_4 = 4 / 255
    eps_8 = 8 / 255

    x_u_4 = (x + torch.FloatTensor(x.size()).uniform_(-eps_4, eps_4).to(args.device)).clamp_(0., 1.)
    x_g_4 = (x + torch.randn(x.size()).clamp_(-eps_4, eps_4).to(args.device)).clamp_(0., 1.)
    x_u_8 = (x + torch.FloatTensor(x.size()).uniform_(-eps_8, eps_8).to(args.device)).clamp_(0., 1.)
    x_g_8 = (x + torch.randn(x.size()).clamp_(-eps_8, eps_8).to(args.device)).clamp_(0., 1.)

    sample_likelihood_dict['uniform_4'] = f_forward(x_u_4, y, 'uniform_4')
    sample_likelihood_dict['uniform_8'] = f_forward(x_u_8, y, 'uniform_8')
    sample_likelihood_dict['gaussian_4'] = f_forward(x_g_4, y, 'gaussian_4')
    sample_likelihood_dict['gaussian_8'] = f_forward(x_g_8, y, 'gaussian_8')

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_2,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_2 = adversary.perturb(x, y)

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_4,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_4 = adversary.perturb(x, y)

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_8,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_8 = adversary.perturb(x, y)

    # adversary = CW(sdim, n_classes, max_iterations=1000, c=1, clip_min=0., clip_max=1., learning_rate=0.01,
    #                targeted=False)
    #
    # adv_cw_1, _, _, _ = adversary.perturb(x, y)
    #
    # adversary = CW(sdim, n_classes, max_iterations=1000, c=10, clip_min=0., clip_max=1., learning_rate=0.01,
    #                targeted=False)
    #
    # adv_cw_10, _, _, _ = adversary.perturb(x, y)

    sample_likelihood_dict['pgd_2'] = f_forward(adv_pgd_2, y, 'pgd_2')
    sample_likelihood_dict['pgd_4'] = f_forward(adv_pgd_4, y, 'pgd_4')
    sample_likelihood_dict['pgd_8'] = f_forward(adv_pgd_8, y, 'pgd_8')
    # sample_likelihood_dict['cw_1'] = f_forward(adv_cw_1, y, 'cw_1')
    # sample_likelihood_dict['cw_10'] = f_forward(adv_cw_10, y, 'cw_10')

    print(sample_likelihood_dict)
    save_dir = hydra.utils.to_absolute_path('attack_logs/case_study')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(sample_likelihood_dict, os.path.join(save_dir, 'sample_likelihood_dict.pt'))


def extract_thresholds(sdim, args):
    sdim.eval()
    # Get thresholds
    threshold_list1 = []

    logger.info("Extracting thresholds ...")
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    for label_id in range(args.get(args.dataset).n_classes):
        # No data augmentation(crop_flip=False) when getting in-distribution thresholds
        dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, label_id=label_id, crop_flip=False)
        in_test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

        in_ll_list = []
        for batch_id, (x, y) in enumerate(in_test_loader):
            x = x.to(args.device)
            y = y.to(args.device)
            ll = sdim(x)

            correct_idx = ll.argmax(dim=1) == y

            ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())

        thresh_idx = int(0.01 * len(in_ll_list))
        thresh1 = sorted(in_ll_list)[thresh_idx]
        threshold_list1.append(thresh1)  # class mean as threshold

    logger.info('thresholds extracted!')
    thresholds1 = torch.tensor(threshold_list1).to(args.device)
    return thresholds1


def adv_eval_with_rejection(sdim, adversary, args, thresholds):
    """
    An attack run with rejection policy.
    """
    sdim.eval()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False, drop_last=True)

    clean_acc_meter = AverageMeter('clean_acc')
    clean_error_meter = AverageMeter('clean_error')
    rejection_meter = AverageMeter('rejection_rate')
    left_acc_meter = AverageMeter('left_acc')
    left_error_meter = AverageMeter('left_error')

    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            output = sdim(x)

        pred = output.argmax(dim=1)
        correct_idx = pred == y
        x, y = x[correct_idx], y[correct_idx]  # Only evaluate on the correct classified samples by clean classifier.
        n_correct = correct_idx.sum().item()

        adv_x = adversary.perturb(x, y)

        with torch.no_grad():
            class_conditionals = sdim(adv_x)

        max_class_conditionals, pred = class_conditionals.max(dim=1)
        successful_idx = pred != y   # idx of successful adversarial examples.
        clean_error = successful_idx.sum().item() / n_correct

        # no rejection
        clean_acc_meter.update(1 - clean_error, n_correct)
        clean_error_meter.update(clean_error, n_correct)

        # filter with thresholds
        reject_idx = max_class_conditionals < thresholds[pred]  # idx of successfully rejected samples.
        left_idx = max_class_conditionals >= thresholds[pred]  # idx of left samples for eval.

        reject_rate = reject_idx.sum().item() / n_correct
        left_pred, left_y = pred[left_idx], y[left_idx]
        left_acc = (left_pred == left_y).sum().item() / n_correct
        left_error = (left_pred != left_y).sum().item() / n_correct

        assert reject_rate + left_acc + left_error == 1
        rejection_meter.update(reject_rate, n_correct)
        left_acc_meter.update(left_acc, n_correct)
        left_error_meter.update(left_error, n_correct)

    logger.info("No rejection, clean acc: {:.4f}, clean error: {:.4f}".format(clean_acc_meter.avg, clean_error_meter.avg))
    logger.info("Rejection, rejection_rate: {:.4f}, left acc: {:.4f}, left error: {:.4f}"
                .format(rejection_meter.avg, left_acc_meter.avg, left_error_meter.avg))
    return clean_acc_meter.avg, clean_error_meter.avg, rejection_meter.avg, left_acc_meter.avg, left_error_meter.avg


def pgd_attack(sdim, args):
    thresholds = extract_thresholds(sdim, args)
    results_dict = {'clean_acc': [], 'clean_error': [], 'reject_rate': [], 'left_acc': [], 'left_error': []}
    eps_list = [2/255, 4/255, 6/255, 8/255, 10/255]
    for eps in eps_list:
        adversary = LinfPGDAttack(
            sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=clip_min,
            clip_max=clip_max, targeted=False)
        logger.info('epsilon = {:.4f}'.format(adversary.eps))

        clean_acc, clean_err, reject_rate, left_acc, left_err = adv_eval_with_rejection(sdim, adversary, args, thresholds)
        results_dict['clean_acc'].append(clean_acc)
        results_dict['clean_error'].append(clean_err)
        results_dict['reject_rate'].append(reject_rate)
        results_dict['left_acc'].append(left_acc)
        results_dict['left_error'].append(left_err)
    torch.save(results_dict, 'adv_eval{}_results.pth'.format(suffix_dict[args.base_type]))


if __name__ == "__main__":
    run()