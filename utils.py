import torch
import os
import numpy as np


# Setup GPU on either M1 or NVIDIA GPU
def setup_gpu(os_type):
    print(f'Setting up GPU on: {os_type}')

    if os_type == 'mac':
        available = torch.backends.mps.is_available()
        built = torch.backends.mps.is_built()
        if available and built:
            print('Using GPU: True')
            device = torch.device('mps')
        else:
            print('Using GPU: False')
            device = None
            
    elif os_type == 'nvidia':
        available = torch.cuda.is_available()
        if available:
            print('Using GPU: True')
            device = torch.device('cuda')
        else:
            print('Using GPU: False')
            device = None
    
    return device


# Generates latent vectors of size
# 2D: (nsamples, latent_size)
def generate_z(nsamples, latent_size, device):
    z = torch.randn(nsamples, latent_size, 1, 1, device=device)
    return z


# Save samples and losses to files
def save_train_data(losses, samples, accuracies, losses_fname, samples_fname, acc_fname, cur_dir):
    losses_fname = os.path.join(cur_dir, losses_fname)
    samples_fname = os.path.join(cur_dir, samples_fname)
    acc_fname = os.path.join(cur_dir, acc_fname)
        
    np.save(samples_fname, samples)
    print(f'Saved samples to file {samples_fname}')

    np.save(losses_fname, losses)
    print(f'Saved losses to file {losses_fname}')

    np.save(acc_fname, accuracies)
    print(f'Saved accuracies to file {acc_fname}')


