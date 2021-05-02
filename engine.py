import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def train_model(train_loader, model, optimizer, criterion, device, lr_scheduler):
    """
    Note: train_loss and train_acc is accurate only if set drop_last=False in loader

    :param train_loader: y: one_hot float tensor
    :param model:
    :param optimizer:
    :param criterion: set reduction='sum'
    :param device:
    :return:
    """
    train_loss = 0
    correct = 0
    batch_start_time = 0
    samples = 0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # target: (B,)
        data, target = data.to(device), target.to(device)
        # (B, C, T)
        logits = model(data)

        # (B, T)
        target_repeat = target.unsqueeze(1).repeat(1, logits.size(-1))
        # (B, T)
        loss = criterion(logits, target_repeat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        with torch.no_grad():
            # (B, C)
            probs = F.softmax(logits, dim=1).mean(-1)
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += target_repeat.numel()

        # batch_start_time = timeit.default_timer()

    train_loss /= samples
    recording_num = samples / float(target_repeat.size(-1))
    train_acc = correct / recording_num
    # train_acc = correct / len(train_loader.dataset)
    return {'loss': train_loss, 'acc': train_acc}


def eval_model(test_loader, model, criterion, device):

    model.eval()
    test_loss = 0
    correct = 0
    samples = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            # target: (B,)
            data, target = data.to(device), target.to(device)
            # (B, C, T)
            logits = model(data)
            # (B, T)
            target_repeat = target.unsqueeze(1).repeat(1, logits.size(-1))
            loss = criterion(logits, target_repeat)
            test_loss += loss.item()
            # get the index of the max log-probability
            # (B, C)
            probs = F.softmax(logits, dim=1).mean(-1)
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += target_repeat.numel()

    test_loss /= samples
    test_acc = correct / len(test_loader.dataset)

    return {'loss': test_loss, 'acc': test_acc}
