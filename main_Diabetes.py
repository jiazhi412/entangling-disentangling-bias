import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
from dataloader.Diabetes_I import DiabetesDataset_I
from dataloader.Diabetes_II import DiabetesDataset_II
from dataloader.Diabetes import DiabetesDataset 
import datetime

from EnD import *
from collections import defaultdict
import models
from tqdm import tqdm
import utils
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_correct(outputs,labels):
    outputs = F.sigmoid(outputs)
    correct = (outputs.round() == labels).sum()
    # print(correct)
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
        bce, abs = criterion(outputs, labels.float(), color_labels.float(), weights)
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
    output_list = []
    target_list = []
    a_list = []
    for data, labels, color_labels in tqdm(dataloader, leave=False):
        data, labels, color_labels = data.to(device), labels.to(device), color_labels.to(device)

        with torch.no_grad():
            outputs = model(data)
            # print(outputs)
        loss = criterion(outputs, labels.float(), color_labels.float(), weights)

        batch_size = data.shape[0]
        tot_correct += num_correct(outputs, labels).item()
        num_samples += batch_size
        # print(tot_correct)
        # print(num_samples)
        # print('daj')
        tot_loss += loss.item() * batch_size

        output_list.append(outputs)
        target_list.append(labels)
        a_list.append(color_labels)

    test_output, test_target, test_a = torch.cat(output_list), torch.cat(target_list), torch.cat(a_list)
    test_acc_p, test_acc_n = utils.compute_subAcc_withlogits_binary(test_output, test_target, test_a)
    D = test_acc_p - test_acc_n
    avg_accuracy = tot_correct / num_samples
    avg_loss = tot_loss / num_samples
    return avg_accuracy, avg_loss, test_acc_p, test_acc_n, D

def main(config):
    # seed = 42
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed)


    opt = vars(config)
    
    print('Note', 'YoungP', 'YoungN', 'OldP', 'OldN')
    # imbalance 
    if opt['bias_type'] == 'I':
        # split 80% for train and 20% for test
        print("Train")
        train_set = DiabetesDataset_I(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    minority=opt['minority'], minority_size=opt['minority_size'], mode='train')
        print("Test")
        test_set = DiabetesDataset_I(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    minority=None, mode='test', balance=True, idx=train_set.get_idx())
        dev_set = test_set

    # association
    elif opt['bias_type'] == 'II':
        print("Train")
        train_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_train_mode'])
        print("Val")
        dev_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_test_mode'], idx=train_set.get_idx())
        print("Test")
        test_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_test_mode'], idx=train_set.get_idx())

    train_loader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  num_workers=config.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=config.num_workers, pin_memory=True)

    print('Training debiased model')
    print('Config:', config)

    in_dim = 8
    hidden_dims = [32, 10]
    model = models.simple_MLP(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=config.n_class)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
    hook = Hook(model.for_hook, backward=False)

    def ce(outputs, labels, color_labels, weights):
        return F.binary_cross_entropy_with_logits(outputs, labels)

    def ce_abs(outputs, labels, color_labels, weights):
        loss = ce(outputs, labels, color_labels, weights)
        abs = abs_regu(hook, labels, color_labels, config.alpha, config.beta)
        return loss, abs

    best = defaultdict(float)

    for i in range(config.epochs):
        train_acc, train_loss, train_bce, train_abs = train(model, train_loader, ce_abs, None, optimizer, scheduler=None)
        scheduler.step()

        valid_acc, valid_loss, test_acc_p, test_acc_n, D = test(model, valid_loader, ce, None)
        # print(valid_acc)

        # biased_test_acc, biased_test_loss = test(model, biased_test_loader, ce, None)
        # unbiased_test_acc, unbiased_test_loss = test(model, unbiased_test_loader, ce, None)

        print(f'Epoch {i} - Train acc: {train_acc:.4f}, train_loss: {train_loss:.4f} (bce: {train_bce:.4f} abs: {train_abs:.4f});')
        print(f'Valid acc {valid_acc:.4f}, loss: {valid_loss:.4f}')
        # print(f'Biased test acc: {biased_test_acc:.4f}, loss: {biased_test_loss:.4f}')
        # print(f'Unbiased test acc: {unbiased_test_acc:.4f}, loss: {unbiased_test_loss:.4f}')

        if valid_acc > best['valid_acc']:
            best = dict(
                valid_acc = valid_acc,
                # biased_test_acc = biased_test_acc,
                # unbiased_test_acc = unbiased_test_acc
            )

    # Output the mean AP for the best model on dev and test set
    if config.bias_type == "I":
        data = {
            'Time': [datetime.datetime.now()],
            'Bias': [opt['bias_attr']],
            'Minority': [opt['minority']],
            'Minority_size': [opt['minority_size']],
            'Test_acc_old': [test_acc_p*100],
            'Test_acc_young': [test_acc_n*100],
            'D': [D*100],
            }
        utils.append_data_to_csv(data, 'Diabetes_EnD_I_trials.csv')
    elif config.bias_type == "II":
        data = {
            'Time': [datetime.datetime.now()],
            'Bias': [opt['bias_attr']],
            'Train': [opt['Diabetes_train_mode']],
            'Test': [opt['Diabetes_test_mode']],
            'Test Acc': [valid_acc * 100]
            }
        utils.append_data_to_csv(data, 'Diabetes_EnD_II_trials.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--color_var', default=0.0, type=float, help='variance for color distribution')
    parser.add_argument('--load_path', default='/nas/vista-ssd01/users/jiazli/datasets/Diabetes/Diabetes_newData.csv')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to resume')
    parser.add_argument('--random_seed', default=1, type=int, help='random seed')
    parser.add_argument('--lr_decay_period', default=3, type=int, help='lr decay period')
    parser.add_argument('--max_step', default=5, type=int, help='maximum step for training')

    parser.add_argument('--n_class', default=1, type=int, help='number of classes')
    parser.add_argument('--n_class_bias', default=1, type=int, help='number of bias classes')
    parser.add_argument('--input_size', default=28, type=int, help='input size')
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='lr decay rate')
    parser.add_argument('--seed', default=1, type=int, help='seed index')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--alpha', default=0.1, type=float, help='EnD alpha')
    parser.add_argument('--beta', default=0.1, type=float, help='EnD beta')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=26)
    parser.add_argument('--local', dest='local', action='store_true', help='disable wandb')    
    parser.add_argument('--rho', type=float, default=0.997, help='rho for biased mnist (.999, .997, .995, .990)')

    parser.add_argument('--log_step', default=150, type=int, help='step for logging in iteration')
    parser.add_argument('--save_step', default=1, type=int, help='step for saving in epoch')
    parser.add_argument('--data_dir', default='/nas/vista-ssd01/users/jiazli/datasets/CMNIST/generated_uniform', help='data directory')
    parser.add_argument('--save_dir', default='./results', help='save directory for checkpoint')
    parser.add_argument('--data_split', default='train', help='data split to use')
    parser.add_argument('--use_pretrain', default=False, type=bool, help='whether it use pre-trained parameters if exists')
    parser.add_argument('--train_baseline', action='store_true', help='whether it train baseline or unlearning')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cudnn_benchmark', default=True, type=bool, help='cuDNN benchmark')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    parser.add_argument('--is_train', default=1, type=int, help='whether it is training')

    ## Census
    parser.add_argument("--bias_attr", type=str, default='age', choices=['sex', 'race', 'age'])
    parser.add_argument("--bias_type", type=str, default='I', choices=['I', 'II', 'General'])

    # Type I Bias
    parser.add_argument("--minority", type=str, default='young')
    parser.add_argument("--minority_size", type=int, default=100)

    # Type II Bias
    parser.add_argument("--Diabetes_train_mode", type=str, default='eb1')
    parser.add_argument("--Diabetes_test_mode", type=str, default='eb2')

    config = parser.parse_args()
    main(config)
