

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from datasets import get_dataloaders
import einops
import lava.lib.dl.slayer as slayer
from sdn import SDN
from utils import apply_noise, normalize_event_frames

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--dataset', type=str,
                        choices=['n-mnist', 'cifar10-dvs', 'dvsgesture'])
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--n_bins', default=5, type=int)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--noise', type=str, choices=[None, 'background', 'hot'], default=None)
    parser.add_argument('--severity', type=int, choices=[1, 2, 3, 4, 5], default=1)
    args = parser.parse_args()
    return args


# batch = 128  # batch size
# lr = 0.01  # leaerning rate
lam = 0.3  # lagrangian for event rate loss
# epochs = 400  # training epochs
steps = [120, 240, 320]  # learning rate reduction milestones


def main():
    args = get_args()
    
    trained_folder = f'experiments/{args.experiment}'
    logs_folder = f'{trained_folder}/Logs'
    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-5)

    # Datasets
    train_loader, val_loader, num_classes = get_dataloaders(
        args.batch_size, args.n_bins, args.dataset, 'data', args.binarize)

    net = SDN(in_channels=2).to(device)

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net=net,
        error=lambda output, target: F.mse_loss(
            output.flatten(), target.flatten()),
        optimizer=optimizer,
        stats=stats,
        count_log=True,
        lam=lam
    )

    for epoch in range(args.epochs):
        if epoch in steps:
            for param_group in optimizer.param_groups:
                print('\nLearning rate reduction from', param_group['lr'])
                param_group['lr'] /= 10/3

        if not args.validate:    
            for i, (input, ground_truth) in enumerate(train_loader):  # training loop
                # BTCHW to BCHWT
                input = input.permute(0, 2, 3, 1)
                ground_truth = einops.repeat(ground_truth, 'batch classes -> batch classes timesteps', timesteps=args.n_bins)
                
                assistant.train(input, ground_truth)
                print(f'\r[Epoch {epoch:3d}/{args.epochs}] {stats}', end='')


        for i, (input, ground_truth) in enumerate(val_loader):  # testing loop
            if not args.binarize:
                input = normalize_event_frames(input)
            
            if args.noise is not None:
                input = apply_noise(input, args.noise, args.severity)
            
            input = input.permute(0, 2, 3, 1)
            ground_truth = einops.repeat(ground_truth, 'batch classes -> batch classes timesteps', timesteps=args.n_bins)
            
            assistant.test(input, ground_truth)
            print(f'\r[Epoch {epoch:3d}/{args.epochs}] {stats}', end='')

        if epoch % 50 == 49:
            print()
        if stats.testing.best_loss:
            torch.save(net.state_dict(), trained_folder + f'/network_ep{epoch}_best.pt')
            
        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/BEST_network.pt')
        stats.update()
        stats.save(trained_folder + '/')

        # gradient flow monitoring
        net.grad_flow(trained_folder + '/')

        # checkpoint saves
        if epoch % 10 == 0:
            torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict(
            )}, logs_folder + f'/checkpoint{epoch}.pt')

    file = open("report.txt", 'a')
    file.write(f"SDN {args.experiment} {args.dataset} binarize={args.binarize} {args.batch_size} {args.lr} {stats.testing.max_accuracy}\n")
    file.close()

if __name__ == "__main__":
    main()