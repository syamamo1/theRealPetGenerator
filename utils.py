import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# Both loss functions take the output of the discriminator

# Loss on real dataset
# smoothing if we want to "generalize more"
# real labels = 1
def real_loss(preds, criterion, smooth=False):
    nsamples = preds.size(0)

    # smooth, real labels = 0.9
    if smooth:
        labels = torch.ones(nsamples)*0.9
    else:
        labels = torch.ones(nsamples)

    loss = criterion(preds.squeeze(), labels)
    return loss

# Loss on generated dataset
# fake labels = 0
def fake_loss(preds, criterion):
    nsamples = preds.size(0)
    labels = torch.zeros(nsamples) 
    loss = criterion(preds.squeeze(), labels)
    return loss


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

