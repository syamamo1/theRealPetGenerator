import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# Generate nsampels latent vectors in 2D tensor
def generate_z(nsamples, latent_size):
    z = np.random.uniform(-1, 1, size=(nsamples, latent_size))
    z = torch.from_numpy(z).float()
    return z


# Plot Gen/Disc losses through epochs
# Input is 2D numpy array where col0 is Gen and col1 is Disc
def plot_losses(losses):
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
    plt.show()


# Save samples and losses to files
def save_train_data(losses, samples):
    samples_fname = 'train_samples.pkl'
    with open(samples_fname, 'wb') as f:
        pkl.dump(samples, f)
        print(f'Saved samples to file {samples_fname}')

    losses_fname = 'train_losses.npy'
    with open(losses_fname, 'wb') as f:
        np.save(f, losses)
        print(f'Saved losses to file {losses_fname}')


# Finish this up!
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    # Load samples from generator, taken while training
    with open('train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)

    rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
    cols = 6
    fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            img = img.detach()
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

