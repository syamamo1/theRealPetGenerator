import torch
import numpy as np
import matplotlib.pyplot as plt

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