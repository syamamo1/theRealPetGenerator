import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from generator import Generator
from discriminator import Discriminator
from utils import setup_gpu, save_train_data, generate_z
from utils import plot_losses, view_samples, print_message
from load_data import load_data


# Ensure sanity
print('Hello World!')

# Global Variables
nchannels = 3
img_size = 64 # height and width of images
latent_size = 100

batch_size = 128 # Number of real images fed to the descriminator per training cycle
num_epochs = 100

print('Num epochs {}, batch size {}'.format(num_epochs, batch_size))

cur_dir = os.getcwd() #  '/ifs/CS/replicated/home/syamamo1/'
data_path = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator', 'dogs-vs-cats')

# Filenames for saved stuff
losses_fname = 'train_losses.npy'
samples_fname = 'train_samples.npy'
acc_fname = 'train_accuracies.npy'


# Runner code!
def main():

    train_mode = 1
    eval_mode = 0

    if train_mode:
        device = setup_gpu('nvidia')

        G = Generator(latent_size, nchannels, device).to(device)
        D = Discriminator(nchannels, device).to(device)

        print('Num trainable Discriminator:', sum(p.numel() for p in D.parameters()))
        print('Num trainable Generator:', sum(p.numel() for p in G.parameters()))
        print('Using {} GPUs'.format(torch.cuda.device_count()))

        d_optimizer = optim.Adam(D.parameters())
        g_optimizer = optim.Adam(G.parameters())

        losses, samples, accuracies = train(D, G, d_optimizer, g_optimizer, latent_size, device)

        # Save results
        save_train_data(losses, samples, accuracies, losses_fname, samples_fname, acc_fname, cur_dir)

    if eval_mode:
        # Visualize results
        plot_losses(losses_fname)
        view_samples(samples_fname)


def train(D, G, d_optimizer, g_optimizer, latent_size, device):
    print('Starting train!')

    # Constant z to track generator change
    num_constant = 4
    constant_z = generate_z(num_constant, latent_size, device)
    
    samples = np.zeros((num_epochs, num_constant, nchannels, img_size, img_size))
    losses = np.zeros((num_epochs, 2))
    accuracies = np.zeros((num_epochs, 2))

    # Train time
    D.train()
    G.train()
    
    for epoch in tqdm(range(num_epochs), desc='Training Models'):
        
        # load data in batches of size batch_size. Weights are updated after predictions are made for every batch
        
        real_data_loader = load_data(data_path=data_path, batch_size=batch_size, img_size=img_size)
        num_batches = 37500//batch_size
        sum_dloss, sum_gloss = 0, 0
        sum_racc, sum_facc = 0, 0
        for batch_num, (real_images, _) in tqdm(enumerate(real_data_loader), desc=f'Epoch {epoch}', total=num_batches):
            real_images = real_images.to(device)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================

            # Update discriminator less bc 
            # performing too well
            update_every = 3
            if batch_num % update_every == 0:
                d_optimizer.zero_grad()
                
                # Get predictions of real images
                preds_real = D(real_images)
                
                # Get predictions of fake images
                z = generate_z(batch_size, latent_size, device)
                fake_images = G(z)
                preds_fake = D(fake_images)
                
                # Compute loss
                d_loss = D.loss(preds_real, preds_fake, smooth=True)
                sum_dloss += d_loss.item()

                # Update discriminator model
                d_loss.backward()
                d_optimizer.step()

            else: sum_dloss += 0
            
            # =========================================
            #            TRAIN GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = generate_z(batch_size, latent_size, device)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake, smooth=True) 
            sum_gloss += g_loss.item()
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()

            # Find discriminator accuracy
            acc_real, acc_fake = D.accuracy(preds_real, preds_fake)
            sum_racc += acc_real
            sum_facc += acc_fake

            # Print some loss stats
            if batch_num % 1 == 0:
                print_message(epoch, num_epochs, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake)
            

        # After each epoch.....

        # Save average accuracies
        accuracies[epoch, 0] += sum_racc/num_batches
        accuracies[epoch, 1] += sum_facc/num_batches

        # Save epoch loss
        losses[epoch, 0] += sum_gloss/num_batches
        losses[epoch, 1] += sum_dloss/(num_batches/update_every)
         
        # Generate and save images
        G.eval() 
        generated_images = G(constant_z)
        samples[epoch] += generated_images.detach().cpu().numpy()
        G.train()

        # print(torch.cuda.memory_summary())

    print('Finished train')
    return losses, samples, accuracies




if __name__ == '__main__':
    main()