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

        # Simple model for now
        self.model = nn.Sequential(
            nn.Conv2d(nchannels, 64, 3, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, bias=False),
            nn.ReLU(True),
            nn.Flatten(1, -1),
            nn.Linear(230400, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, 1, bias=False),
            nn.Sigmoid()
        )

        # Layer dictionary!
        # nn.BatchNorm2d(input_channels)
        # nn.MaxPool2d(2)
        # nn.Dropout(0.3)
    

    # Sum of real, fake loss
    def loss(self, real_preds, fake_preds, smooth=False):
        real_loss = self.real_loss(real_preds, smooth)
        fake_loss = self.fake_loss(fake_preds)
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
    def fake_loss(self, preds):
        nsamples = preds.size(0)
        labels = zeros(nsamples, device = self.device) 
        loss = self.criterion(preds.squeeze(), labels)
        return loss
    

    # Forward pass
    def forward(self, x):
        return self.model(x)