import torch
import os
import numpy as np
import socket
import torch.nn as nn
# import torch.distributed as dist


# Setup GPU on either M1 or NVIDIA GPU
def setup_gpu(config):
    world_size = torch.cuda.device_count()

    print(f'Setting up GPU on: {config.gpu_type}')
    print('Using {} GPUs'.format(world_size))

    # MAC
    if config.gpu_type == 'mac':
        available = torch.backends.mps.is_available()
        if available:
            print('Using GPU: True')
            device = torch.device('mps')
        else:
            print('Using GPU: False')
            device = None
            
    # NVIDIA
    elif config.gpu_type == 'nvidia':
        available = torch.cuda.is_available()
        if available:
            print('Using GPU: True')

            if config.multiple_gpus:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ["MASTER_PORT"] = str(get_free_port())
                device = None

            else:
                device = torch.device('cuda')
            
        else:
            print('Using GPU: False')
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
def generate_zeros2(config, world_size):
    nzeros = config.batch_size//world_size
    d_losses, g_losses = np.zeros(nzeros), np.zeros(nzeros)
    real_accs, fake_accs = np.zeros(nzeros), np.zeros(nzeros)
    realpreds, fakepreds = np.zeros(nzeros), np.zeros(nzeros)
    return d_losses, g_losses, real_accs, fake_accs, realpreds, fakepreds


# Save samples and losses to files

def save_train_data(config, cur_dir, losses, samples, accuracies, av_preds):
    losses_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', config.losses_fname)
    samples_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', config.samples_fname)
    acc_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', config.acc_fname)
    av_preds_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', config.av_preds_fname)
        
    np.save(samples_fname, samples)
    print(f'Saved samples to file {samples_fname}')

    np.save(losses_fname, losses)
    print(f'Saved losses to file {losses_fname}')

    np.save(acc_fname, accuracies)
    print(f'Saved accuracies to file {acc_fname}')

    np.save(av_preds_fname, av_preds)
    print(f'Saved average predictions to file {av_preds_fname}')


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

