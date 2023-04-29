import torch.nn as nn
from torch import ones, zeros

# Binary classifier (real or fake)
# Outputs the PROBABILITY that an image is REAL
# Takes in an image
class Discriminator(nn.Module):
    def __init__(self, config, device=None):
        super(Discriminator, self).__init__()

        # yaml
        self.config = config

        # Binary cross entropy: evaluates probabilities
        self.criterion = nn.BCELoss() 

        # GPU device!
        self.device = device

        # Killer model now
        self.model = nn.Sequential(
            nn.Conv2d(config.nchannels, 64, kernel_size=4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # Layer dictionary!
        # nn.BatchNorm2d(input_channels)
        # nn.MaxPool2d(2)
        # nn.Dropout(0.3)
    

    # Sum of real, fake loss
    def loss(self, real_preds, fake_preds):
        real_loss = self.real_loss(real_preds)
        fake_loss = self.fake_loss(fake_preds)
        loss = real_loss + fake_loss
        return loss


    # These are real images so their labels are 1
    # smoothing if we want to "generalize more"
    def real_loss(self, preds):
        nsamples = preds.size(0)
        labels = ones(nsamples, device = self.device) * self.config.real_labels

        loss = self.criterion(preds.squeeze(), labels)
        return loss
    

    # These are fake images so their labels are 0
    def fake_loss(self, preds):
        nsamples = preds.size(0)
        labels = ones(nsamples, device = self.device) * self.config.fake_labels

        loss = self.criterion(preds.squeeze(), labels)
        return loss
    

    # Forward pass
    def forward(self, x):
        return self.model(x)
    
    # Num correct/Num samples
    def accuracy(self, preds_real, preds_fake):
        correct_real = (preds_real > 0.5).sum().item()
        correct_fake = (preds_fake < 0.5).sum().item()

        acc_real = correct_real/preds_real.size(0)
        acc_fake = correct_fake/preds_fake.size(0)

        return acc_real, acc_fake