import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

import pickle as pkl

from generator import Generator
from discriminator import Discriminator
from utils import real_loss, fake_loss
from load_data import load_data

def main():
    # instantiate discriminator and generator
    D = Discriminator()
    G = Generator()
    # check that they are as you expect
    print(D)
    print(G)

    lr = 0.002
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)

    data = load_data()
    train(D, G, d_optimizer, g_optimizer, data)

def train(D, G, d_optimizer, g_optimizer, data):
    # training hyperparams
    num_epochs = 100

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 400

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    D.train()
    G.train()
    for epoch in range(num_epochs):
        
        for batch_i, (real_images, _) in enumerate(train_loader):
                    
            batch_size = real_images.size(0)
            
            ## Important rescaling step ## 
            real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
            
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            
            d_optimizer.zero_grad()
            
            # 1. Train with real images

            # Compute the discriminator losses on real images 
            # smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)
            
            # 2. Train with fake images
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images        
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)
            
            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            
            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # 1. Train with fake images and flipped labels
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake) # use real loss to flip labels
            
            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        
        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to train mode


    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
