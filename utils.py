import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from load_data import load_data

# Generates latent vectors of size
# 2D: (nsamples, latent_size)
def generate_z(nsamples, latent_size, device):
    z = torch.randn(nsamples, latent_size, 1, 1, device=device)
    return z


# Save samples and losses to files
def save_train_data(losses, samples, losses_fname, samples_fname):
    with open(samples_fname, 'wb') as f:
        pkl.dump(samples, f)
        print(f'Saved samples to file {samples_fname}')

    np.save(losses_fname, losses)
    print(f'Saved losses to file {losses_fname}')


# Plot Gen/Disc losses through epochs
# Input is 2D numpy array where col0 is Gen and col1 is Disc
def plot_losses(fname):
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

    with open(samples_fname, 'rb') as f:
        samples = pkl.load(f)

    # Use this for epochs = 100
    every_n_epochs = 10 
    cols = 4
    _, axes = plt.subplots(figsize=(7,12), nrows=every_n_epochs, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::int(len(samples)/every_n_epochs)], axes):
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
