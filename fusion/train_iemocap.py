from __future__ import print_function

import argparse
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from model import LMF
from utils import total, load_iemocap
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

import shutil
from pyutils.config import configs
from pyutils.torch_train import set_torch_deterministic
from core.utils.logging import logger as lg

def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))

def main(options):

    configs.run_dir = os.path.join(
            "./tt_runs",
            # f'TT_ATTN_{configs.model.TT_ATTN}',
            # f'TT_FUSION_{configs.model.TT_FUSION}',
            # f'TT_SUBNET_{configs.model.TT_SUBNET}',
            # time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
            str(os.getpid())
        )
    os.makedirs(configs.run_dir, exist_ok=True)
    # shutil.copy(args.config, configs.run_dir)
    run_dir_config_path = os.path.join(configs.run_dir, "config.yml")  
    os.makedirs(os.path.dirname(run_dir_config_path), exist_ok=True)
    shutil.copyfile(configs.config_dir, run_dir_config_path)
    
    # grid search
    configs.output_path = os.path.join(configs.run_dir, "grid_search.csv")  
    os.makedirs(os.path.dirname(configs.output_path), exist_ok=True)
    with open(configs.output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow([str(os.getpid()),])
        # writer.writerow(['batch_size','factor_learning_rate', 'learning_rate', 'weight_decay', 'Test Accuracy Score', 'Test F1-score'])
    
    lg.init(configs)
    
    DTYPE = torch.FloatTensor
    LONG = torch.LongTensor
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    # parse the input args
    epochs = configs.run.n_epochs
    data_path = configs.dataset.dataset_dir
    patience = configs.run.patience
    emotion = configs.dataset.emotion
    output_dim = configs.model.output_dim
    
    # epochs = options['epochs']
    # data_path = options['data_path']
    # model_path = options['model_path']
    # output_path = options['output_path']
    # signiture = options['signiture']
    # patience = options['patience']
    # emotion = options['emotion']
    # output_dim = options['output_dim']


    # prepare the paths for storing models and outputs
    # model_path = os.path.join(
    #     model_path, "model_{}_{}.pt".format(signiture, emotion))
    # output_path = os.path.join(
    #     output_path, "results_{}_{}.csv".format(signiture, emotion))
    # print("Temp location for models: {}".format(model_path))
    # print("Grid search results are in: {}".format(output_path))
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_set, valid_set, test_set = load_iemocap(data_path, emotion)

    params = dict()
    # params['audio_hidden'] = [8, 16, 32]
    # params['video_hidden'] = [4, 8, 16]
    # params['text_hidden'] = [64, 128, 256]
    # params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    # params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    # params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    # params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    # params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    # params['rank'] = [1, 4, 8, 16]
    # params['batch_size'] = [8, 16, 32, 64, 128]
    # params['weight_decay'] = [0, 0.001, 0.002, 0.01]
    
    # params['audio_hidden'] = [8,]
    # params['video_hidden'] = [4,]
    # params['text_hidden'] = [64,]
    # params['audio_dropout'] = [0.3,]
    # params['video_dropout'] = [0.1,]
    # params['text_dropout'] = [0.15,]
    # params['rank'] = [8,]
    
    params['batch_size'] = [configs.dataset.batch_size, ] if type(configs.dataset.batch_size) is not list else configs.dataset.batch_size
    params['factor_learning_rate'] = [configs.optimizer.factor_lr, ] if type(configs.optimizer.factor_lr) is not list else configs.optimizer.factor_lr
    params['learning_rate'] = [configs.optimizer.lr, ] if type(configs.optimizer.lr) is not list else configs.optimizer.lr
    params['weight_decay'] = [configs.optimizer.weight_decay, ] if type(configs.optimizer.weight_decay) is not list else configs.optimizer.weight_decay
    
    params['random_state'] = [configs.run.random_state, ] if type(configs.run.random_state) is not list else configs.run.random_state
    params['phase_bias'] = [configs.noise.phase_bias, ] if type(configs.noise.phase_bias) is not list else configs.noise.phase_bias
    params['gamma_noise_std'] = [configs.noise.gamma_noise_std, ] if type(configs.noise.gamma_noise_std) is not list else configs.noise.gamma_noise_std
    params['crosstalk_factor'] = [configs.noise.crosstalk_factor, ] if type(configs.noise.crosstalk_factor) is not list else configs.noise.crosstalk_factor

    import itertools
    # total_settings = total(params)
    keys, values = zip(*params.items())
    total_settings = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("There are {} different hyper-parameter settings in total.".format(len(total_settings)))

    # seen_settings = set()

    # if not os.path.isfile(output_path):
    #     with open(output_path, 'w+') as out:
    #         writer = csv.writer(out)
    #         writer.writerow(["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
    #                         'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay', 
    #                         'Best Validation CrossEntropyLoss', 'Test CrossEntropyLoss', 'Test F1-score', 'Test Accuracy Score'])

    # for i in range(total_settings):
    for i_set, setting in enumerate(total_settings):
        # print(setting)
        factor_lr = setting['factor_learning_rate']
        lr = setting['learning_rate']
        batch_sz = setting['batch_size']
        decay = setting['weight_decay']
        
        configs.run.random_state = setting['random_state']
        configs.noise.phase_bias = setting['phase_bias']
        configs.noise.gamma_noise_std = setting['gamma_noise_std']
        configs.noise.crosstalk_factor = setting['crosstalk_factor']

        # ahid = random.choice(params['audio_hidden'])
        # vhid = random.choice(params['video_hidden'])
        # thid = random.choice(params['text_hidden'])
        # thid_2 = thid // 2
        # adr = random.choice(params['audio_dropout'])
        # vdr = random.choice(params['video_dropout'])
        # tdr = random.choice(params['text_dropout'])
        # factor_lr = random.choice(params['factor_learning_rate'])
        # lr = random.choice(params['learning_rate'])
        # r = random.choice(params['rank'])
        # batch_sz = random.choice(params['batch_size'])
        # decay = random.choice(params['weight_decay'])

        # # reject the setting if it has been tried
        # current_setting = (ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay)
        # if current_setting in seen_settings:
        #     continue
        # else:
        #     seen_settings.add(current_setting)

        # model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
        # if options['cuda']:
        #     model = model.cuda()
        #     DTYPE = torch.cuda.FloatTensor
        #     LONG = torch.cuda.LongTensor
        
        
        ### Reset seed for each sub run
        if hasattr(configs.run, "random_state"):
            set_torch_deterministic(configs.run.random_state) 
        else:
            configs.run.random_state = 42
            set_torch_deterministic(configs.run.random_state)
        
        from utils import build_fusion_model
        model = build_fusion_model(device=device)
        model = model.to(device)
        
        if i_set == 0:
            lg.info(model)
            lg.info(str(os.getpid()))
        
        print("Model initialized")
        criterion = nn.CrossEntropyLoss(size_average=False)
        # factors = list(model.parameters())[:3]
        # other = list(model.parameters())[3:]
        
        factors = list()
        other = list()
        for name, param in model.named_parameters():
            if "factor" in name:
                # print(name)
                factors.append(param)
            else:
                other.append(param)
        
        optimizer = optim.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay)

        # setup training
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
        valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
        test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
        curr_patience = patience
        
        # test before training
        model.eval()
        for batch in test_iterator:
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
            y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False).to(device)
            output = model(x_a, x_v, x_t)
            valid_loss = criterion(output, torch.max(y, 1)[1])
            avg_valid_loss = valid_loss.item()
        y = y.cpu().data.numpy().reshape(-1, output_dim)

        if np.isnan(avg_valid_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        avg_valid_loss = avg_valid_loss / len(valid_set)
        # print("Validation loss is: {}".format(avg_valid_loss))
        
        all_true_label = np.argmax(y,axis=1)
        all_predicted_label = np.argmax(output.detach().cpu(),axis=1)
        
        avg_valid_accuracy = accuracy_score(all_true_label, all_predicted_label)
        f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
        # print("Validation F1 is: {}".format(f1))
        lg.info("Epoch: 0, Train loss: {:.4f}, Train acc: {:.4f}, Val loss: {:.4f}, Val acc: {:.4f}, Val F1: {:.4f}".format(avg_valid_loss, avg_valid_accuracy, avg_valid_loss, avg_valid_accuracy, f1))
        
        if args.test_only:
            lg.info(f"random_state: {configs.run.random_state}, phase_bias:{configs.noise.phase_bias}, gamma_noise_std: {configs.noise.gamma_noise_std}, crosstalk_factor:{configs.noise.crosstalk_factor}" + "Test acc: {:.4f}, Test F1: {:.4f}".format(avg_valid_accuracy, f1))

            with open(configs.output_path, 'a+') as out:
                writer = csv.writer(out)
                # writer.writerow([configs.run.random_state, configs.noise.phase_bias, configs.noise.gamma_noise_std, configs.noise.crosstalk_factor, round(avg_valid_accuracy, 4), round(f1,4)])
                writer.writerow([round(f1,4)])

        else:
            for e in range(epochs):
                ##### Training #####
                model.train()
                model.zero_grad()
                avg_train_loss = 0.0
                for batch in train_iterator:
                    model.zero_grad()

                    x = batch[:-1]
                    x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
                    x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
                    x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False).to(device)
                    try:
                        output = model(x_a, x_v, x_t)
                    except ValueError as e:
                        print(x_a.data.shape)
                        print(x_v.data.shape)
                        print(x_t.data.shape)
                        raise e
                    loss = criterion(output, torch.max(y, 1)[1])
                    loss.backward()
                    avg_loss = loss.item()
                    avg_train_loss += avg_loss / len(train_set)
                    optimizer.step()
                
                    all_true_label = np.concatenate((all_true_label, np.argmax( y.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0)
                    all_predicted_label = np.concatenate((all_predicted_label,np.argmax(output.cpu().data.numpy().reshape(-1, output_dim),axis=1)), axis=0) 
                
                avg_train_accuracy = accuracy_score(all_true_label, all_predicted_label)

                # print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

                # Terminate the training process if run into NaN
                if np.isnan(avg_train_loss):
                    print("Training got into NaN values...\n\n")
                    complete = False
                    break

                ##### Validation #####
                model.eval()
                for batch in valid_iterator:
                    x = batch[:-1]
                    x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
                    x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
                    x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False).to(device)
                    output = model(x_a, x_v, x_t)
                    valid_loss = criterion(output, torch.max(y, 1)[1])
                    avg_valid_loss = valid_loss.item()
                y = y.cpu().data.numpy().reshape(-1, output_dim)

                if np.isnan(avg_valid_loss):
                    print("Training got into NaN values...\n\n")
                    complete = False
                    break

                avg_valid_loss = avg_valid_loss / len(valid_set)
                # print("Validation loss is: {}".format(avg_valid_loss))
                
                all_true_label = np.argmax(y,axis=1)
                all_predicted_label = np.argmax(output.detach().cpu(),axis=1)
                avg_valid_accuracy = accuracy_score(all_true_label, all_predicted_label)
                f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
                # print("Validation F1 is: {}".format(f1))

                if (avg_valid_loss < min_valid_loss):
                    curr_patience = patience
                    min_valid_loss = avg_valid_loss
                    # torch.save(model, model_path)
                    torch.save(model.state_dict(), os.path.join(configs.run_dir, "best_model.pt"))
                    # print("Found new best model, saving to disk...")
                else:
                    curr_patience -= 1
                
                if curr_patience <= 0:
                    break
                # print("\n")
                
                lg.info("Epoch: {}, Train loss: {:.4f}, Train acc: {:.4f}, Val loss: {:.4f}, Val acc: {:.4f}, Val F1: {:.4f}".format(e+1, avg_train_loss, avg_train_accuracy, avg_valid_loss, avg_valid_accuracy, f1))

            if complete:
                
                model.load_state_dict(torch.load(os.path.join(configs.run_dir, "best_model.pt")))
                model.eval()
                for batch in test_iterator:
                    x = batch[:-1]
                    x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).to(device)
                    x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).to(device)
                    x_t = Variable(x[2].float().type(DTYPE), requires_grad=False).to(device)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False).to(device)
                    output_test = model(x_a, x_v, x_t)
                    loss_test = criterion(output_test, torch.max(y, 1)[1])
                    test_loss = loss_test.item()
                output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
                y = y.cpu().data.numpy().reshape(-1, output_dim)
                test_loss = test_loss / len(test_set)

                # these are the needed metrics
                all_true_label = np.argmax(y,axis=1)
                all_predicted_label = np.argmax(output_test,axis=1)

                f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
                acc_score = accuracy_score(all_true_label, all_predicted_label)

                display(f1, acc_score)
                lg.info(f"Param bz: {batch_sz}, factor_lr:{factor_lr}, lr: {lr}, weight_devcay:{decay}" + "Test acc: {:.4f}, Test F1: {:.4f}".format(acc_score, f1))

                with open(configs.output_path, 'a+') as out:
                    writer = csv.writer(out)
                    # writer.writerow([batch_sz, round(factor_lr, 4), round(lr, 4), round(decay, 3), round(acc_score, 4), round(f1,4)])
                    writer.writerow([round(f1,4),])


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    # OPTIONS.add_argument('--emotion', dest='emotion', type=str, default='angry')
    # OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    # OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    # OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    # OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    # OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    # OPTIONS.add_argument('--data_path', dest='data_path',
    #                      type=str, default='data/')
    # OPTIONS.add_argument('--model_path', dest='model_path',
    #                      type=str, default='models')
    # OPTIONS.add_argument('--output_path', dest='output_path',
    #                      type=str, default='results')
    
    OPTIONS.add_argument("config", metavar="FILE", help="config file")
    OPTIONS.add_argument('--test_only', action='store_true', default=False, help="test")
    args, opts = OPTIONS.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.config_dir = args.config
    configs.update(opts)
    
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)