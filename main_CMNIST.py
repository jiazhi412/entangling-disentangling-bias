import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
import CMNIST

from EnD import *
from collections import defaultdict
import models
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_correct(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    correct = preds.eq(labels).sum()
    return correct

def train(model, dataloader, criterion, weights, optimizer, scheduler):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0
    tot_bce = 0.
    tot_abs = 0.
    model.train()

    for data, labels, color_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device), labels.to(device), color_labels.to(device)

        optimizer.zero_grad()
        with torch.enable_grad():
            outputs = model(data)
        bce, abs = criterion(outputs, labels, color_labels, weights)
        loss = bce+abs
        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size
        tot_bce += bce.item() * batch_size
        tot_abs += abs.item() * batch_size

    if scheduler is not None:
        scheduler.step()

    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss, tot_bce/num_samples, tot_abs/num_samples

def test(model, dataloader, criterion, weights):
    num_samples = 0
    tot_correct = 0
    tot_loss = 0

    model.eval()

    for data, labels, color_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device), labels.to(device), color_labels.to(device)

        with torch.no_grad():
            outputs = model(data)
        loss = criterion(outputs, labels, color_labels, weights)

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        tot_loss += loss.item() * batch_size

    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss

def main(config):
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    custom_loader = CMNIST.WholeDataLoader(config, istrain=True)
    train_loader = torch.utils.data.DataLoader(custom_loader,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=config.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(custom_loader,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=config.num_workers, pin_memory=True)

    custom_loader_test = CMNIST.WholeDataLoader(config, istrain=False)
    biased_test_loader = torch.utils.data.DataLoader(custom_loader_test,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)    
    unbiased_test_loader = torch.utils.data.DataLoader(custom_loader_test,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)    


    print('Training debiased model')
    print('Config:', config)

    model = models.simple_convnet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    hook = Hook(model.avgpool, backward=False)

    def ce(outputs, labels, color_labels, weights):
        return F.cross_entropy(outputs, labels)

    def ce_abs(outputs, labels, color_labels, weights):
        loss = ce(outputs, labels, color_labels, weights)
        abs = abs_regu(hook, labels, color_labels, config.alpha, config.beta)
        return loss, abs

    best = defaultdict(float)

    for i in range(config.epochs):
        train_acc, train_loss, train_bce, train_abs = train(model, train_loader, ce_abs, None, optimizer, scheduler=None)
        scheduler.step()

        valid_acc, valid_loss = test(model, valid_loader, ce, None)
        biased_test_acc, biased_test_loss = test(model, biased_test_loader, ce, None)
        unbiased_test_acc, unbiased_test_loss = test(model, unbiased_test_loader, ce, None)

        print(f'Epoch {i} - Train acc: {train_acc:.4f}, train_loss: {train_loss:.4f} (bce: {train_bce:.4f} abs: {train_abs:.4f});')
        print(f'Valid acc {valid_acc:.4f}, loss: {valid_loss:.4f}')
        print(f'Biased test acc: {biased_test_acc:.4f}, loss: {biased_test_loss:.4f}')
        print(f'Unbiased test acc: {unbiased_test_acc:.4f}, loss: {unbiased_test_loss:.4f}')

        if valid_acc > best['valid_acc']:
            best = dict(
                valid_acc = valid_acc,
                biased_test_acc = biased_test_acc,
                unbiased_test_acc = unbiased_test_acc
            )

    import datetime
    import pandas as pd
    def append_data_to_csv(data,csv_name):
        df = pd.DataFrame(data)
        if os.path.exists(csv_name):
            df.to_csv(csv_name,mode='a',index=False,header=False)
        else:
            df.to_csv(csv_name,index=False)
    import datetime
    data = {
        'Time': [datetime.datetime.now()],
        'Var': [config.color_var],
        'End': [biased_test_acc * 100]
        }
    # append_data_to_csv(data, os.path.join(self.exp_dir, 'CMNIST_End_trials.csv'))
    append_data_to_csv(data, 'CMNIST_End_trials.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_name', default='csad0020', help='experiment name')
    parser.add_argument('--color_var', default=0.0, type=float, help='variance for color distribution')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to resume')
    parser.add_argument('--random_seed', default=1, type=int, help='random seed')
    parser.add_argument('--lr_decay_period', default=3, type=int, help='lr decay period')
    parser.add_argument('--max_step', default=5, type=int, help='maximum step for training')

    parser.add_argument('--n_class', default=10, type=int, help='number of classes')
    parser.add_argument('--n_class_bias', default=8, type=int, help='number of bias classes')
    parser.add_argument('--input_size', default=28, type=int, help='input size')
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='lr decay rate')
    parser.add_argument('--seed', default=1, type=int, help='seed index')
    # parser.add_argument('--alpha', default=1, type=int, help='alpha')
    parser.add_argument('--tau', default=10, type=int, help='tau')
    parser.add_argument('--lambda_', default=1, type=int, help='lambda')
    # parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--alpha', default=0.1, type=float, help='EnD alpha')
    parser.add_argument('--beta', default=0.1, type=float, help='EnD beta')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--local', dest='local', action='store_true', help='disable wandb')    
    parser.add_argument('--rho', type=float, default=0.997, help='rho for biased mnist (.999, .997, .995, .990)')

    parser.add_argument('--log_step', default=150, type=int, help='step for logging in iteration')
    parser.add_argument('--save_step', default=1, type=int, help='step for saving in epoch')
    parser.add_argument('--data_dir', default='/nas/vista-ssd01/users/jiazli/datasets/CMNIST/generated_uniform', help='data directory')
    parser.add_argument('--save_dir', default='./results', help='save directory for checkpoint')
    parser.add_argument('--data_split', default='train', help='data split to use')
    parser.add_argument('--use_pretrain', default=False, type=bool,
                        help='whether it use pre-trained parameters if exists')
    parser.add_argument('--train_baseline', action='store_true', help='whether it train baseline or unlearning')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cudnn_benchmark', default=True, type=bool, help='cuDNN benchmark')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    parser.add_argument('--is_train', default=1, type=int, help='whether it is training')


    config = parser.parse_args()
    print(config)
    main(config)
