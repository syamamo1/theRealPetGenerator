import torch.nn as nn
import torch.nn.functional as F


# Binary classifier (real or fake)
# Takes in an image
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ninput_channels = 3
        
        # Need to update this!
        self.model = [
            # Block 1
            nn.Conv2d(ninput_channels, 64),
            nn.Conv2d(64, 64),
            nn.Conv2d(64, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(128, 128),
            nn.Conv2d(128, 128),
            nn.Conv2d(128, 128),
            nn.BatchNorm2d(128),
            
            # dropout layer 
            nn.Dropout(0.3)
        ]
    
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x