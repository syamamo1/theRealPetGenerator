import torch.nn as nn
import torch.nn.functional as F


# Binary classifier (real or fake)
# Takes in an image
class Discriminator(nn.Module):
    def __init__(self, nchannels, height, width):
        super(Discriminator, self).__init__()

        # Simple model for now
        # Input: (n, input_channels, height, width) images
        # Output: 1D binary (n) - probability that sample is REAL
        self.model = nn.Sequential(
            nn.Conv2d(nchannels, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.Flatten(1, -1),
            nn.Linear(64*height*width, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Layer dictionary!
        # nn.BatchNorm2d(input_channels)
        # nn.MaxPool2d(2)
        # nn.Dropout(0.3)
    
    
    def forward(self, x):
        return self.model(x)