# Ensure sanity
print('Hello World!')

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from easydict import EasyDict
import torch.multiprocessing as mp
import torch.distributed as dist

from generator import Generator
from discriminator import Discriminator
from utils import setup_gpu, save_train_data, generate_z, generate_zeros1, generate_zeros2, weights_init
from visualizers import plot_losses, view_samples, plot_accuracy, plot_together, view_dataset
from logs import print_message, print_pytorch_stats, log_message, log_time, newfile
from logs import log_extra
from load_data import load_data


local = 0

# Set shtuff up
cur_dir = os.getcwd() #  '/ifs/CS/replicated/home/syamamo1/'
working_dir = os.path.join(cur_dir, 'course', 'cs1430', 'theRealPetGenerator')
config_path = 'config.yaml'
data_path = 'dogs-vs-cats'

# Remote
if not local: 
    data_path = os.path.join(working_dir, data_path) 
    config_path = os.path.join(working_dir, config_path) 

# Load config file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)


# Runner code!
def main():

    # Train mode
    if config.train_mode:
        print(config)
        device, world_size = setup_gpu(config)
        print_pytorch_stats(config)

        if config.multiple_gpus:          
            print(f'Spawning on {world_size} GPUs!')
            newfile(config)
            mp.spawn(train, args=(config, world_size), nprocs=world_size) 

        else:
            print('No spawning here!')
            train(device, config, world_size)
            

    # Visualize results
    if config.eval_mode:
        # plot_losses(config)
        # plot_accuracy(config)
        # plot_together(config)
        view_samples(config)
        view_dataset(config, data_path)


# Rank is device (torch.cuda/torch.mps) for 1GPU
# or 1...n for multi GPU
def train(rank, config, world_size):
    # Load data
    data_loader = load_data(config, data_path, world_size, rank)

    # Load models
    if config.multiple_gpus:
        print(f'Starting train! Rank {rank}')
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        G = Generator(config, rank).to(rank)
        D = Discriminator(config, rank).to(rank)
        G.model = DDP(G.model, device_ids=[rank], broadcast_buffers=False)
        D.model = DDP(D.model, device_ids=[rank], broadcast_buffers=False)  

    else:
        print('Starting train!')
        G = Generator(config, rank).to(rank)
        D = Discriminator(config, rank).to(rank)

    # Randomly initialize weights to N(0, 0.02)
    G.apply(weights_init)
    D.apply(weights_init)
        
    # Load optimizers
    d_optimizer = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Constant z to track generator change
    constant_z = generate_z(config, config.num_constant, rank)
    samples, losses, accuracies, av_preds = generate_zeros1(config)

    # Train time
    D.train()
    G.train()

    # Iterate epochs
    num_batches = len(data_loader.dataset)//(world_size*config.batch_size)
    for epoch in tqdm(np.arange(1, config.num_epochs+1), desc='Training Models'):
        if not config.multiple_gpus: print(f'Starting epoch {epoch}')
        
        # Iterate batches
        d_losses, g_losses, real_accs, fake_accs, realpreds, fakepreds = generate_zeros2(config, world_size)
        start_time = datetime.datetime.now()
        for batch_num, (real_images, _) in enumerate(data_loader, 1):
            real_images = real_images.to(rank)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================

            # Update discriminator less bc 
            # performing too well
            if batch_num % config.update_every == 1:
                d_optimizer.zero_grad()
                
                # Get predictions of real images
                preds_real = D(real_images)
                
                # Get predictions of fake images
                z = generate_z(config, config.batch_size, rank)
                fake_images = G(z)
                preds_fake = D(fake_images)
                
                # Compute loss
                d_loss = D.loss(preds_real, preds_fake)

                # Update discriminator model
                d_loss.backward()
                d_optimizer.step()
            
            # =========================================
            #            TRAIN GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = generate_z(config, config.batch_size, rank)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake) 
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()

            # Save some stats
            av_real_pred = torch.mean(preds_real).item()
            av_fake_pred = torch.mean(preds_fake).item()
            realpreds[batch_num] += av_real_pred
            fakepreds[batch_num] += av_fake_pred

            d_losses[batch_num] += d_loss.item()
            g_losses[batch_num] += g_loss.item()

            acc_real, acc_fake = D.accuracy(preds_real, preds_fake)
            real_accs[batch_num] += acc_real
            fake_accs[batch_num] += acc_fake

            # Print some loss/pred/accuracy stats
            if batch_num % config.print_every == 0:
                if config.multiple_gpus:
                    # dist.barrier()
                    log_message(config, rank, epoch, batch_num, num_batches, 
                                d_loss, g_loss, acc_real, acc_fake)
                    info = acc_real, acc_fake, preds_real, preds_fake, av_real_pred, av_fake_pred
                    log_extra(config, rank, epoch, batch_num, info)
                else:
                    print_message(epoch, config.num_epochs, batch_num, 
                                num_batches, d_loss, g_loss, acc_real, acc_fake)


        # After each epoch.....

        # Save average prediction
        av_preds[epoch-1, 0] += np.mean(realpreds)
        av_preds[epoch-1, 1] += np.mean(fakepreds)

        # Save average accuracies
        accuracies[epoch-1, 0] += np.mean(real_accs)
        accuracies[epoch-1, 1] += np.mean(fake_accs)

        # Save epoch loss
        losses[epoch-1, 0] += np.mean(g_losses)
        losses[epoch-1, 1] += np.mean(d_losses)
         
        # Generate and save images
        G.eval() 
        with torch.no_grad():
            generated_images = G(constant_z)
            samples[epoch] += generated_images.detach().cpu().numpy()
        G.train()

        # print(torch.cuda.memory_summary())
        if config.multiple_gpus: log_time(config, epoch, start_time)
        else: print(f'Train time for epoch {epoch}:', datetime.datetime.now()-start_time)

    # DONE-ZO
    print('Finished train')
    save_train_data(config, cur_dir, losses, samples, accuracies, av_preds)




if __name__ == '__main__':
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print("Total time:", elapsed_time)