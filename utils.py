import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from load_data import load_data
from tqdm import tqdm


# Print messages about training
def print_message(epoch, num_epochs, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake):
    message = '''
               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] 
                  acc_real: {:6.4f} | d_loss: {:6.4f}
                  acc_fake: {:6.4f} | g_loss: {:6.4f}  
                '''.format(
                        epoch, num_epochs, batch_num, num_batches, 
                        acc_real, d_loss.item(), 
                        acc_fake, g_loss.item())
    tqdm.write(message)


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
    losses_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', losses_fname)
    samples_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', samples_fname)
    acc_fname = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', acc_fname)
        
    np.save(samples_fname, samples)
    print(f'Saved samples to file {samples_fname}')

    np.save(losses_fname, losses)
    print(f'Saved losses to file {losses_fname}')

    np.save(acc_fname, accuracies)
    print(f'Saved accuracies to file {acc_fname}')


# Plot Gen/Disc losses through epochs
# Input is 2D numpy array where col0 is Gen and col1 is Disc
def plot_losses(fname):
    print(f'Plotting losses from: {fname}')
    losses = np.load(fname)

    num_epochs = losses.shape[0]
    x = np.arange(num_epochs)
    gen = losses[:, 0]
    disc = losses[:, 1]

    plt.figure()
    plt.plot(x, gen, label='Generator Loss')
    plt.plot(x, disc, label='Discriminator Loss')
    plt.title('Model Losses vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Binary Cross Entropy)')
    plt.legend()
    plt.show()


# View generator results though epochs
def view_samples(samples_fname):
    print(f'Viewing samples from: {samples_fname}')

    with open(samples_fname, 'rb') as f:
        samples = pkl.load(f)

    # Use this for epochs = 100
    every_n_epochs = 3
    cols = 4
    rows = int(len(samples)/every_n_epochs)
    _, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::every_n_epochs], axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            img = img.detach().cpu()
            ax.imshow(img.permute(1,2,0))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    # Use this for viewing first 5 epochs
    # rows = 5
    # cols = 4
    # _, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    # for sample, ax_row in zip(samples, axes):
    #     for img, ax in zip(sample, ax_row):
    #         img = img.detach().cpu()
    #         ax.imshow(img.permute(1,2,0))
    #         ax.xaxis.set_visible(False)
    #         ax.yaxis.set_visible(False)

    plt.show()

        