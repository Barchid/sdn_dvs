import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import lava.lib.dl.slayer as slayer

# from pilotnet_dataset import PilotNetDataset
# import utils
torch.manual_seed(4205)

def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


class SDN(torch.nn.Module):
    def __init__(self, in_channels: int, dataset: str = "n-mnist"):
        super(SDN, self).__init__()
        
        sdnn_params = { # sigma-delta neuron parameters
                'threshold'     : 0.1,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : True,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
                'activation'    : F.relu, # activation function
            }
        sdnn_cnn_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
            }
        sdnn_dense_params = { # dense layers have additional dropout units enabled
                **sdnn_cnn_params,                        # copy all sdnn_cnn_params
                'dropout' : slayer.neuron.Dropout(p=0.2), # neuron dropout
            }
        
        
        if dataset == 'n-mnist':
            self.blocks = self._nmnist_architecture(sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels)
        elif dataset == 'cifar10-dvs':
            self.blocks = self._cifar10dvs_architecture(sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels)
            
        else:
            self.blocks = self._dvsgesture_architecture(sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels)
            
        # self.blocks = torch.nn.ModuleList([# sequential network blocks 
        #         # delta encoding of the input
        #         slayer.block.sigma_delta.Input(sdnn_params), 
        #         # convolution layers
        #         slayer.block.sigma_delta.Conv(sdnn_cnn_params,  3, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
        #         slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
        #         slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
        #         slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
        #         # flatten layer
        #         slayer.block.sigma_delta.Flatten(),
        #         # dense layers
        #         slayer.block.sigma_delta.Dense(sdnn_dense_params, 64*40, 100, weight_scale=2, weight_norm=True),
        #         slayer.block.sigma_delta.Dense(sdnn_dense_params,   100,  50, weight_scale=2, weight_norm=True),
        #         slayer.block.sigma_delta.Dense(sdnn_dense_params,    50,  10, weight_scale=2, weight_norm=True),
        #         # linear readout with sigma decoding of output
        #         slayer.block.sigma_delta.Output(sdnn_dense_params,   10,   1, weight_scale=2, weight_norm=True)
        #     ])
        
    def _nmnist_architecture(self, sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels):
        return torch.nn.ModuleList([
            # delta encoding of the input
            slayer.block.sigma_delta.Input(sdnn_params),
            
            # conv + pool
            # 1st block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_channels, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 2nd block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # flatten layer
            slayer.block.sigma_delta.Flatten(),
            
            # FC
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 10368, 10368//4, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   10368//4,  100, weight_scale=2, weight_norm=True),
            
            # linear readout with sigma decoding of output
            slayer.block.sigma_delta.Output(sdnn_dense_params,   100,   10, weight_scale=2, weight_norm=True)
        ])
        
    def _cifar10dvs_architecture(self, sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels):
        return torch.nn.ModuleList([
            # delta encoding of the input
            slayer.block.sigma_delta.Input(sdnn_params),
            
            # conv + pool
            # 1st block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_channels, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 2nd block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 3d
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 4th
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # flatten layer
            slayer.block.sigma_delta.Flatten(),
            
            # FC
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 8192, 8192//4, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   8192//4,  100, weight_scale=2, weight_norm=True),
            
            # linear readout with sigma decoding of output
            slayer.block.sigma_delta.Output(sdnn_dense_params,   100,   10, weight_scale=2, weight_norm=True)
        ])
        
    def _dvsgesture_architecture(self, sdnn_params, sdnn_cnn_params, sdnn_dense_params, in_channels):
        return torch.nn.ModuleList([
            # delta encoding of the input
            slayer.block.sigma_delta.Input(sdnn_params),
            
            # conv + pool
            # 1st block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_channels, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 2nd block
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 3d
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 4th
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # 5th
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 128, 128, 3, padding=1, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Pool(sdnn_cnn_params, 2, stride=2, weight_scale=2, weight_norm=True),
            
            # flatten layer
            slayer.block.sigma_delta.Flatten(),
            
            # FC
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 2048, 2048//4, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   2048//4,  100, weight_scale=2, weight_norm=True),
            
            # linear readout with sigma decoding of output
            slayer.block.sigma_delta.Output(sdnn_dense_params,   100,   11, weight_scale=2, weight_norm=True)
        ])
        
    
    def forward(self, x):
        count = []
        event_cost = 0

        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum(torch.abs((x[..., 1:] > 0).to(x.dtype))).item())

        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
    
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
        