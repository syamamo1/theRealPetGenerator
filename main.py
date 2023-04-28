# Ensure sanity
print('Hello World!')

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
from utils import setup_gpu, save_train_data, generate_z, generate_zeros
from visualizers import plot_losses, view_samples, plot_accuracy, plot_together, view_dataset
from logs import print_message, print_pytorch_stats, log_message, log_time, newfile
from logs import log_extra
from load_data import load_data


local = True

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
print(config)


# Runner code!
def main():

    # Train mode
    if config.train_mode:
        device, world_size = setup_gpu(config)

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
        # view_samples(config)
        view_dataset(config, data_path)


def train(rank, config, world_size):
    print('Starting train!')
    if config.multiple_gpus:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        print('Rank:', dist.get_rank())
        G = Generator(config.latent_size, config.nchannels, rank).to(rank)
        D = Discriminator(config.nchannels, rank).to(rank)
        G.model = DDP(G.model, device_ids=[rank], broadcast_buffers=False)
        D.model = DDP(D.model, device_ids=[rank], broadcast_buffers=False)  
        print_pytorch_stats(G, D)

    else:
        G = Generator(config.latent_size, config.nchannels, rank).to(rank)
        D = Discriminator(config.nchannels, rank).to(rank)
        print_pytorch_stats(G, D)

    d_optimizer = optim.Adam(D.parameters())
    g_optimizer = optim.Adam(G.parameters())

    # Constant z to track generator change
    constant_z = generate_z(config, config.num_constant, rank)
    samples, losses, accuracies = generate_zeros(config)

    # Train time
    D.train()
    G.train()

    # Iterate epochs
    num_batches = config.dataset_size//(world_size*config.batch_size)
    for epoch in tqdm(range(config.num_epochs), desc='Training Models'):
        print(f'Starting epoch {epoch}')
        
        # Iterate batches
        real_data_loader = load_data(config, data_path, world_size, rank)
        sum_dloss, sum_gloss, sum_racc, sum_facc = 0, 0, 0, 0,
        start_time = datetime.datetime.now()
        for batch_num, (real_images, _) in enumerate(real_data_loader):
            real_images = real_images.to(rank)
            
            # ============================================
            #            TRAIN DISCRIMINATOR
            # ============================================

            # Update discriminator less bc 
            # performing too well
            if batch_num % config.update_every == 0:
                d_optimizer.zero_grad()
                
                # Get predictions of real images
                preds_real = D(real_images)
                
                # Get predictions of fake images
                z = generate_z(config, config.batch_size, rank)
                fake_images = G(z)
                preds_fake = D(fake_images)
                
                # Compute loss
                d_loss = D.loss(preds_real, preds_fake, config.smooth)
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
            z = generate_z(config, config.batch_size, rank)
            fake_images = G(z)
            
            # Compute loss
            preds_fake = D(fake_images)
            g_loss = G.loss(preds_fake, config.smooth) 
            sum_gloss += g_loss.item()
            
            # Update generator model
            g_loss.backward()
            g_optimizer.step()

            # Find discriminator accuracy
            acc_real, acc_fake = D.accuracy(preds_real, preds_fake)
            sum_racc += acc_real
            sum_facc += acc_fake

            # Print some loss/accuracy stats
            if batch_num % config.print_every == 0:
                if config.multiple_gpus:
                    # dist.barrier()
                    log_message(config, rank, epoch, batch_num, 
                                num_batches, d_loss, g_loss, acc_real, acc_fake)
                    info = acc_real, acc_fake, preds_real, preds_fake
                    log_extra(config, rank, epoch, batch_num, info)
                else:
                    print_message(epoch, config.num_epochs, batch_num, 
                                num_batches, d_loss, g_loss, acc_real, acc_fake)
            

        # After each epoch.....

        # Save average accuracies
        accuracies[epoch, 0] += sum_racc/num_batches
        accuracies[epoch, 1] += sum_facc/num_batches

        # Save epoch loss
        losses[epoch, 0] += sum_gloss/num_batches
        losses[epoch, 1] += sum_dloss/(num_batches/config.update_every)
         
        # Generate and save images
        G.eval() 
        generated_images = G(constant_z)
        samples[epoch] += generated_images.detach().cpu().numpy()
        G.train()

        # print(torch.cuda.memory_summary())
        if config.multiple_gpus: log_time(config, epoch, start_time)
        else: print(f'Train time for epoch {epoch}:', datetime.datetime.now()-start_time)


    # DONE-ZO
    print('Finished train')
    save_train_data(config, losses, samples, accuracies, cur_dir)




if __name__ == '__main__':
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print("Total time:", elapsed_time)