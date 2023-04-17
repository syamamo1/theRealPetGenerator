import torch.nn as nn
import torch.nn.functional as F


# Takes in a 1D latent vector (noise)
# Outputs a real-lookin' photo of a cat/dog
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # 1
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        # 2
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        
        # 3
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 4
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # 5
        out = F.tanh(self.fc4(x))
        return out