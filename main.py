import argparse
import torch
import os
import torch.optim as optim
import glob
from engine import *
from utils.check_point import CheckPoint
from utils.history import History
from utils.utilities import get_logger
import numpy as np
import random
import logging
from torch.utils.data import DataLoader
from layers.base import LabelSmoothingLoss
from models import ARCH_REGISTRY
from data import DATA_REGISTRY
from data.data_transformer import RandomCropWav, FakePitchShift, Compose, TimeStretch
from data.sampler import BalancedSampler
import csv
import pprint
import socket
import yaml
from layers.base import FakeLrScheduler


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_next_run(exp_dir):
    next_run = 0
    files = glob.glob(os.path.join(exp_dir, "*.log"))
    for file in files:
        filename = os.path.basename(file)
        run = filename.split('.')[0]
        id = int(run[3:])
        if id >= next_run:
            next_run = id

    next_run += 1
    return 'Run{:03d}'.format(next_run)


def run(args):

    set_seed(args.seed)
    exp_dir = '{}/ckpt/{}/'.format(ROOT_DIR, args.exp)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if args.ckpt_prefix == 'auto':
        args.ckpt_prefix = get_next_run(exp_dir)

    # setup logging info
    log_file = '{}/{}.log'.format(exp_dir, args.ckpt_prefix)
    logger = get_logger(log_file)
    logger.info("IP:{}".format(get_local_ip()))
    logger.info('\n'+pprint.pformat(vars(args)))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    dataset_cls = DATA_REGISTRY.get(args.dataset)
    time_stretch_args = eval(args.time_stretch_args)
    transform = Compose([
        FakePitchShift(target_sr=args.sr, pitch_shift_steps=eval(args.pitch_shift_steps)),
        RandomCropWav(target_sr=args.sr, crop_seconds=args.crop_seconds) if args.crop_seconds > 0 else None,
        TimeStretch(target_sr=args.sr, stretch_args=time_stretch_args) if time_stretch_args[0] > 0 else None
    ])
    train_set = dataset_cls(fold=args.fold,
                            split='train',
                            target_sr=args.sr,
                            transform=transform,)
    train_loader = DataLoader(dataset=train_set,
                              sampler=BalancedSampler(dataset=train_set,
                                                      labels=train_set.get_train_labels(),
                                                      shuffle=True),
                              batch_size=args.batch_size, drop_last=True, num_workers=8)
    val_loader = DataLoader(dataset=dataset_cls(fold=args.fold,
                                                split='valid',
                                                target_sr=args.sr,),
                            batch_size=32, drop_last=False, shuffle=False, num_workers=8)

    model_cls = ARCH_REGISTRY.get(args.net)
    model = model_cls(**vars(args)).to(device)

    logger.info(model)

    for param in model.feat.parameters():
        param.requires_grad = False

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.init_lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.init_lr, weight_decay=args.l2)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.init_lr, weight_decay=args.l2)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.init_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.run_epochs,
                                                       pct_start=getattr(args, 'lr_pct_start', 0.08),
                                                       cycle_momentum=getattr(args, 'lr_cycle_momentum', False),
                                                       div_factor=getattr(args, 'lr_div_factor', 25.0),
                                                       anneal_strategy=getattr(args, 'lr_anneal', 'cos'))
    if not getattr(args, 'lr_schedule_enable', True):
        lr_scheduler = FakeLrScheduler()

    train_hist, val_hist = History(name='train'), History(name='val')

    # checkpoint after new History, order matters
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/{}'.format(ROOT_DIR, args.exp),
                        prefix=args.ckpt_prefix, interval=1, save_num=1, fake_save=getattr(args, 'fake_save', False))

    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)

    from torch.utils.tensorboard import SummaryWriter
    train_writer = SummaryWriter('{}/ckpt/{}/{}/{}'.format(ROOT_DIR, args.exp, args.ckpt_prefix, 'train'))
    valid_writer = SummaryWriter('{}/ckpt/{}/{}/{}'.format(ROOT_DIR, args.exp, args.ckpt_prefix, 'valid'))

    for epoch in range(1, args.run_epochs+1):
        train_hist.add(
            logs=train_model(train_loader, model, optimizer, criterion, device, lr_scheduler),
            epoch=epoch
        )
        val_hist.add(
            logs=eval_model(val_loader, model, criterion, device),
            epoch=epoch
        )

        train_writer.add_scalar("loss", train_hist.recent['loss'], epoch)
        train_writer.add_scalar("acc", train_hist.recent['acc'], epoch)
        valid_writer.add_scalar("loss", val_hist.recent['loss'], epoch)
        valid_writer.add_scalar("acc", val_hist.recent['acc'], epoch)
        train_writer.add_scalar("lr", get_lr(optimizer), epoch)

        # plotting
        if args.plot:
            train_hist.clc_plot()
            val_hist.plot()

        # logging
        logger.info("Epoch{:04d},{:6},{}".format(epoch, train_hist.name, str(train_hist.recent)))
        logger.info("Epoch{:04d},{:6},{}".format(epoch, val_hist.name, str(val_hist.recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist.recent)

    args_results = {**vars(args), **get_best_result(train_hist, val_hist)}
    write_exp_results(result=args_results)
    # explicitly save last
    ckpter.save(epoch=args.run_epochs-1, monitor='acc', loss_acc=val_hist.recent)
    train_writer.close()
    valid_writer.close()


def write_exp_results(result):
    exp = result['exp']
    file_name = '{}/ckpt/{}/{}.csv'.format(ROOT_DIR, exp, exp)
    file_exists = os.path.isfile(file_name)
    if file_exists:
        reader = csv.DictReader(open(file_name, 'r'))
        exist_fieldnames = sorted(reader.fieldnames)
    with open(file_name, 'a+') as f:
        fieldnames = sorted(result.keys())
        if file_exists:
            if not fieldnames == exist_fieldnames:
                logging.info("Writing results, ignored key: {}".format(set(fieldnames) - set(exist_fieldnames)))
                fieldnames = exist_fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def get_best_result(train_hist, val_hist):
    result = {}
    i = np.argmax(val_hist.acc)
    result['train_acc'] = round(train_hist.acc[i], 4)
    result['train_loss'] = round(train_hist.loss[i], 4)
    result['valid_acc'] = round(val_hist.acc[i], 4)
    result['valid_loss'] = round(val_hist.loss[i], 4)
    result['epoch_best'] = train_hist.epoch[i]
    return result


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_cfg(cfg_file):
    configs_dict = yaml.full_load(open(cfg_file, 'r'))
    cfg = dict()
    for k, v in configs_dict.items():
        cfg[k] = v[0]
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfgs/esc/esc_folds_baseline.yaml", type=str)
    args = parser.parse_args()
    cfg = get_cfg(args.cfg)
    run(argparse.Namespace(**cfg))