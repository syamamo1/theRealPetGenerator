import torch.nn as nn
from torch import ones, zeros

# Binary classifier (real or fake)
# Outputs the PROBABILITY that an image is REAL
# Takes in an image
class Discriminator(nn.Module):
    def __init__(self, nchannels, device):
        super(Discriminator, self).__init__()

        # Binary cross entropy: evaluates probabilities
        self.criterion = nn.BCELoss() 

        # GPU device!
        self.device = device

        # Killer model now
        self.model = nn.Sequential(
            nn.Conv2d(nchannels, 64, 3, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(64, 32, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),

            nn.Flatten(1, -1),
            nn.Linear(25088, 128, bias=False),
            nn.LeakyReLU(),

            nn.Linear(128, 32, bias=False),
            nn.LeakyReLU(),

            nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )

        # Setup for multiple GPUs
        # self.model = nn.DataParallel(self.model)

        # Layer dictionary!
        # nn.BatchNorm2d(input_channels)
        # nn.MaxPool2d(2)
        # nn.Dropout(0.3)
    

    # Sum of real, fake loss
    def loss(self, real_preds, fake_preds, smooth=False):
        real_loss = self.real_loss(real_preds, smooth)
        fake_loss = self.fake_loss(fake_preds, smooth)
        loss = real_loss + fake_loss
        return loss


    # These are real images so their labels are 1
    # smoothing if we want to "generalize more"
    def real_loss(self, preds, smooth=False):
        nsamples = preds.size(0)

        # smooth, real labels = 0.9
        if smooth:
            labels = ones(nsamples, device = self.device)*0.9
        else:
            labels = ones(nsamples, device = self.device)

        loss = self.criterion(preds.squeeze(), labels)
        return loss
    

    # These are fake images so their labels are 0
    def fake_loss(self, preds, smooth=False):
        nsamples = preds.size(0)

        # smooth, fake labels = 0.1
        if smooth:
            labels = zeros(nsamples, device = self.device) + 0.1
        else:
            labels = zeros(nsamples, device = self.device)

        loss = self.criterion(preds.squeeze(), labels)
        return loss
    

    # Forward pass
    def forward(self, x):
        return self.model(x)
    

    def accuracy(self, preds_real, preds_fake):
        correct_real = (preds_real > 0.5).sum().item()
        correct_fake = (preds_fake < 0.5).sum().item()

        acc_real = correct_real/preds_real.size(0)
        acc_fake = correct_fake/preds_fake.size(0)

        return acc_real, acc_fake