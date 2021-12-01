# -*- coding: utf-8 -*-
# @Time  : 2021/11/30 14:24
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : baseline.py
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.pardir))
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.optim import Adam
from tqdm import tqdm

from example.dataset import AudioDataset
from example.utils import load_model, save_model
from example.models.Restormer import Restormer
from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming


'''
Train  for the Task1 of the L3DAS22 challenge.
This script saves the model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task1.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def evaluate(model, device, criterion, dataloader):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            outputs = model(x, device)
            loss = criterion(outputs, target)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current val loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    train_dataset = AudioDataset(args, 'L3DAS22_Task1_train100')
    dev_dataset = AudioDataset(args, 'L3DAS22_Task1_dev')
    test_dataset = AudioDataset(args, 'L3DAS22_Task2_dev')


    print('\nShapes:')
    print('Training predictors: ', len(train_dataset))
    print('Validation predictors: ', len(dev_dataset))
    print('Test predictors: ', len(test_dataset))


    #build data loader from dataset
    tr_data = utils.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(dev_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    #LOAD MODEL
    if args.architecture == 'fasnet':
        model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'tac':
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'MIMO_UNet_Beamforming':
        model = MIMO_UNet_Beamforming(fft_size=args.fft_size,
                                      hop_size=args.hop_size,
                                      input_channel=args.input_channel)
    elif args.architecture == 'restormer':
        model = Restormer(fft_size=args.fft_size,
                          hop_size=args.hop_size,
                          inp_channels=args.input_channel)

    print(model)

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total/1e6))
    if args.use_cuda:
        print("Moving model to gpu")

    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    #set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    #load model checkpoint if desired
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda)

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    epoch = 1
    while state["worse_epochs"] < args.patience:
        print("Training epoch " + str(epoch))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(train_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                target = target.to(device)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs = model(x, device)
                loss = criterion(outputs, target)
                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                pbar.set_description("Current train loss: {:.4f}".format(train_loss))
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion, val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            valid_loss = val_loss.cpu().detach().numpy()
            #checkpoint_name = ('%03d' % epoch) + '_' + ('%.6f' % valid_loss) + '.pth'
            #checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint")

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())
            epoch += 1
    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], args.use_cuda)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion, tr_data)
    val_loss = evaluate(model, device, criterion, val_data)
    test_loss = evaluate(model, device, criterion, test_data)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #saving parameters
    parser.add_argument('--results_path', type=str, default='/home/zlf/L3DAS22/result',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/zlf/L3DAS22/checkpoint',
                        help='Folder to write checkpoints into')
    parser.add_argument('--input_dir', default='/home/dataset/L3DAS22', help='the path of L3DAS22 dataset')
    parser.add_argument('--num_mics', default=1)
    parser.add_argument('--sample_rate', default=16000)

    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    # #model parameters
    # parser.add_argument('--architecture', type=str, default='MIMO_UNet_Beamforming',
    #                     help="model name")
    parser.add_argument('--architecture', type=str, default='restormer',
                        help="model name")
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=1)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--fft_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=4)

    args = parser.parse_args()

    #eval string bools
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)

    main(args)
