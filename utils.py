import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dvs_noises import background_activity, hot_pixels
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize_event_frames(input):
    # shape : BTCHW
    batch_size = input.shape[0]
    n_bins = input.shape[1]

    for i in range(batch_size):
        for j in range(n_bins):
            input[i, j, 0, :, :] = input[i, j, 0, :, :] / \
                (input[i, j, 0, :, :].max() + 1e-9)
            input[i, j, 1, :, :] = input[i, j, 1, :, :] / \
                (input[i, j, 1, :, :].max() + 1e-9)

    return input


def apply_noise(input, noise, severity):
    # shape BTCHW
    output = torch.zeros_like(input)
    batch_size = input.shape[0]
    input = input.cpu().numpy()

    for i in range(batch_size):
        event_frame = input[i]

        if noise == 'background':
            output[i] = torch.from_numpy(
                background_activity(event_frame, severity))
        else:
            output[i] = torch.from_numpy(hot_pixels(event_frame, severity)).to(device)
            
            
    return output
