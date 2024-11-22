'''
Description:
Author: Yequan Zhao (yequan_zhao@ucsb.edu)
Date: 2023-04-20 16:07:22
LastEditors: Yequan Zhao (yequan_zhao@ucsb.edu)
LastEditTime: 2023-04-20 17:06:18
'''
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
import sys 
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import platform
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pyutils.config import configs

from core import builder
# from pyutils.general import logger as lg
from core.utils.logging import logger as lg
from pyutils.torch_train import BestKModelSaver, count_parameters, get_learning_rate, set_torch_deterministic
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

import time
import shutil
from utils import build_fusion_model, build_fusion_dataloader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

DTYPE = torch.FloatTensor
LONG = torch.LongTensor

def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    avg_loss = 0.0
    all_true_label = np.array([])
    all_predicted_label = np.array([])

    output_dim = configs.model.output_dim

    for batch_idx, batch in enumerate(train_loader):
        x = batch[:-1]
        x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
        x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
        x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
        y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False).to(device)
        try:
            output = model(x_a, x_v, x_t)
        except ValueError as e:
            print(x_a.data.shape)
            print(x_v.data.shape)
            print(x_t.data.shape)
            raise e
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data.item() / len(train_loader)
        all_true_label = np.concatenate((all_true_label, np.argmax( y.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0)
        all_predicted_label = np.concatenate((all_predicted_label,np.argmax(output.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0)    

        # Terminate the training process if run into NaN
        if np.isnan(avg_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        step += 1

    scheduler.step()
    avg_acc_score = accuracy_score(all_true_label, all_predicted_label)
    lg.info("Epoch: {}, Training loss: {:.2f}, Accuracy: {:.2f}".format(epoch, avg_loss, avg_acc_score))


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    device
) -> None:
    model.eval()
    avg_loss = 0.0
    all_true_label = np.array([])
    all_predicted_label = np.array([])

    output_dim = configs.model.output_dim

    with torch.no_grad():
        for batch in validation_loader:
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
            y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False).to(device)
            output = model(x_a, x_v, x_t)
            loss = criterion(output, y)

            avg_loss += loss.data.item() / len(validation_loader)
            all_true_label = np.concatenate((all_true_label, np.argmax( y.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0)
            all_predicted_label = np.concatenate((all_predicted_label,np.argmax(output.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0) 
    avg_acc_score = accuracy_score(all_true_label, all_predicted_label)
    f1 = f1_score(all_true_label, all_predicted_label, average='weighted')

    lg.info("Epoch: {}, Validation loss: {:.2f}, F1: {:.2f}".format(epoch, avg_loss, f1))
    return avg_loss, avg_acc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    set_torch_deterministic(configs.dataset.seed) 
    
    configs.run_dir = os.path.join(
            "./tt_runs",
            # f'TT_ATTN_{configs.model.TT_ATTN}',
            # f'TT_FUSION_{configs.model.TT_FUSION}',
            # f'TT_SUBNET_{configs.model.TT_SUBNET}',
            time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
        )
    os.makedirs(configs.run_dir, exist_ok=True)
    shutil.copy(args.config, configs.run_dir)
    
    lg.init(configs)
    lg.info(str(os.getpid()))

    model = build_fusion_model()
    model = model.to(device)
    print(model)
    train_loader, validation_loader, test_loader = build_fusion_dataloader()
    factors = list(model.parameters())[:3]
    other = list(model.parameters())[3:]
    optimizer = torch.optim.Adam([{"params": factors, "lr": configs.optimizer.factor_lr}, {"params": other, "lr": configs.optimizer.lr}], weight_decay=configs.optimizer.weight_decay)
    
    # optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    
    ### init accuracy
    avg_valid_loss, avg_valid_accuracy = validate(model, validation_loader, 0, criterion, device)

    lossv, accv = [], []
    epoch = 0
    best_valid_accuracy = 0

    for epoch in range(int(configs.run.n_epochs)):
        train(model, train_loader, optimizer, scheduler, epoch, criterion, device)
        avg_valid_loss, avg_valid_accuracy = validate(model, validation_loader, epoch, criterion, device)

        if (avg_valid_accuracy > best_valid_accuracy):
            best_valid_accuracy = avg_valid_accuracy
            # torch.save(model, os.path.join(configs.run_dir, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(configs.run_dir, "best_model.pt"))

if __name__ == "__main__":
    main()