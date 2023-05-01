import torch
import os
import numpy as np
import socket
import torch.nn as nn
from logs import log
import math
# import torch.distributed as dist


# Setup GPU on either M1 or NVIDIA GPU
def setup_gpu(config):
    world_size = torch.cuda.device_count()

    log(config, f'Setting up GPU on: {config.gpu_type}')
    log(config, 'Using {} NVIDIA GPUs'.format(world_size))

    # MAC
    if config.gpu_type == 'mac':
        available = torch.backends.mps.is_available()
        if available:
            log(config, 'Using GPU: True')
            device = torch.device('mps')
        else:
            log(config, 'Using GPU: False')
            device = None
            
    # NVIDIA
    elif config.gpu_type == 'nvidia':
        available = torch.cuda.is_available()
        if available:
            log(config, 'Using GPU: True')

            if config.multiple_gpus:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ["MASTER_PORT"] = str(get_free_port())
                device = None

            else:
                device = torch.device('cuda')
            
        else:
            log(config, 'Using GPU: False')
            device = None 
    
    return device, world_size


# Generates latent vectors of size
# 2D: (nsamples, latent_size)
def generate_z(config, nsamples, device):
    z = torch.randn(nsamples, config.latent_size, 1, 1, device=device)
    return z


# Return zeros to store stuff in
def generate_zeros1(config):
    samples = np.zeros((config.num_epochs, config.num_constant, config.nchannels, config.img_size, config.img_size))
    losses = np.zeros((config.num_epochs, 2))
    accuracies = np.zeros((config.num_epochs, 2))
    av_preds = np.zeros((config.num_epochs, 2))
    return samples, losses, accuracies, av_preds


# Return zeros to store stuff in, 2
def generate_zeros2(num_batches):
    d_losses, g_losses = np.zeros(num_batches), np.zeros(num_batches)
    real_accs, fake_accs = np.zeros(num_batches), np.zeros(num_batches)
    realpreds, fakepreds = np.zeros(num_batches), np.zeros(num_batches)
    return d_losses, g_losses, real_accs, fake_accs, realpreds, fakepreds


# Save samples and losses to files
def save_train_data(config, cwd, losses, samples, accuracies, av_preds):
    losses_fname = os.path.join(cwd, config.losses_fname)
    samples_fname = os.path.join(cwd, config.samples_fname)
    acc_fname = os.path.join(cwd, config.acc_fname)
    av_preds_fname = os.path.join(cwd, config.av_preds_fname)
        
    np.save(samples_fname, samples)
    log(config, f'Saved samples to file {samples_fname}')

    np.save(losses_fname, losses)
    log(config, f'Saved losses to file {losses_fname}')

    np.save(acc_fname, accuracies)
    log(config, f'Saved accuracies to file {acc_fname}')

    np.save(av_preds_fname, av_preds)
    log(config, f'Saved average predictions to file {av_preds_fname}')


# Save model for loading in future
def save_models(config, G, D, cwd):
    g_path = os.path.join(cwd, config.g_fname)
    d_path = os.path.join(cwd, config.d_fname)
    
    torch.save(G.model, g_path)
    log(config, 'Saved Generator model to:', g_path)

    torch.save(D.model, d_path)
    log(config, 'Saved Discriminator model to:', d_path)
    

# Load trained model
def load_models(config, G, D, cwd):
    g_path = os.path.join(cwd, config.g_fname)
    d_path = os.path.join(cwd, config.d_fname)
    
    G.model = torch.load(g_path)
    log(config, 'Loaded Generator model from:', g_path)
    
    D.model = torch.load(d_path)
    log(config, 'Loaded Discriminator model from:', g_path)

    return G, D



# Thanks kento
def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Initialize weights from N(0, 0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
