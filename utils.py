import torch
import torch.nn as nn

# Both loss functions take the output of the discriminator

# Loss on real dataset
# smoothing if we want to "generalize more"
# real labels = 1
def real_loss(preds, smooth=False):
    nsamples = preds.size(0)

    # smooth, real labels = 0.9
    if smooth:
        labels = torch.ones(nsamples)*0.9
    else:
        labels = torch.ones(nsamples) 

    criterion = nn.BCELoss()
    loss = criterion(preds.squeeze(), labels)
    return loss

# Loss on generated dataset
# fake labels = 0
def fake_loss(preds):
    nsamples = preds.size(0)
    labels = torch.zeros(nsamples) 
    criterion = nn.BCELoss()
    loss = criterion(preds.squeeze(), labels)
    return loss