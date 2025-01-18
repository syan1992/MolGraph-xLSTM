import math
import numpy as np
from typing import Dict, Union, List, Set

import torch
import torch.optim as optim
from torch import nn
from torch_geometric.data import Data

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args: Dict, optimizer: optim.Optimizer, epoch: int, lr: float):
    """Learning rate adjustment methods.

    Args:
        args (Dict): Parsed arguments.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch.
        lr (float): The value of the learning rate.
    """
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer[0].param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(
    opt: Dict[str, Union[str, float, int, List]],
    epoch: int,
    batch_id: int,
    total_batches: int,
    optimizer: optim.Optimizer,
):
    """Learning rate warmup method.

    Args:
        opt (Dict[str,Union[str,float,int,List]]): Parse arguments.
        epoch (int): Current epoch.
        batch_id (int): The number of the current batch.
        total_batches (int): The number of total batch.
        optimizer (Optimizer): Optimizer.
    """
    if opt.warm and epoch <= opt.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (opt.warm_epochs * total_batches)
        lr = opt.warmup_from + p * (opt.warmup_to - opt.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def set_optimizer(lr: float, weight_decay: float, model: nn.Sequential):
    """Initialize the optimizer.

    Args:
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        model (nn.Sequential): Model.

    Returns:
        _type_: _description_
    """
    optimizer = []
    optimizer.append(optim.Adam([{'params':model.parameters()}], lr = lr, weight_decay=weight_decay))
    return optimizer


def calmean(dataset: Set[Data]):
    """Calculate the mean value and the standard deviation value for a regression task.

    Args:
        dataset (Set[Data]): Train set of the regression task.

    Returns:
        The mean value and the standard deviation value of the dataset.
    """
    block_size = 1000
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i].y)

    labels_tensor = torch.Tensor(labels).to("cuda")
    mm = torch.mean(labels_tensor)
    ss = torch.std(labels_tensor)
    yy = (labels_tensor - mm) / ss

    num_samples = len(labels)
    weight_blocks = []

    for start in range(0, num_samples, block_size):
        end = min(start + block_size, num_samples)
        yy_block = yy[start:end].unsqueeze(1)
        weight_block = torch.cdist(yy_block, yy.unsqueeze(1))
        weight_blocks.append(weight_block)

    weight = torch.cat(weight_blocks, dim=0)
    all_weights = []
    for start in range(0, weight.shape[0], block_size):
        end = min(start + block_size, weight.shape[0])
        block = weight[start:end].flatten()
        all_weights.append(block)

    flattened_weights = torch.cat(all_weights)
    median = torch.median(flattened_weights)
    dynamic_t = median
    max_dist = yy.max() - yy.min()

    return mm, ss, dynamic_t, max_dist


def save_model(
    model: nn.Sequential,
    optimizer: optim.Optimizer,
    opt: Dict[str, Union[str, float, int, List]],
    epoch: int,
    save_file: str,
):
    """Save the model.

    Args:
        save_file (str): The address to save the model.
    """

    print("==> Saving...")
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer[0].state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state
