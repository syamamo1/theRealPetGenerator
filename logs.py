import torch
from tqdm import tqdm
import datetime


# Print num parameters, num GPUs
def print_pytorch_stats(G, D):
    print('Num trainable Discriminator:', sum(p.numel() for p in D.parameters()))
    print('Num trainable Generator:', sum(p.numel() for p in G.parameters()))


# Print messages about training
def print_message(epoch, num_epochs, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake):
    message = '''
               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] 
                  acc_real: {:6.4f} | d_loss: {:6.4f}
                  acc_fake: {:6.4f} | g_loss: {:6.4f}  
                '''.format(
                        epoch, num_epochs, batch_num, num_batches, 
                        acc_real, d_loss.item(), 
                        acc_fake, g_loss.item())
    tqdm.write(message)


# Write message to file
def log_message(config, rank, epoch, batch_num, num_batches, d_loss, g_loss, acc_real, acc_fake):
    message = '''
               Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] --> Rank {}
                  acc_real: {:6.4f} | d_loss: {:6.4f}
                  acc_fake: {:6.4f} | g_loss: {:6.4f}  
                '''.format(
                        epoch, config.num_epochs, batch_num, num_batches, rank,
                        acc_real, d_loss.item(), 
                        acc_fake, g_loss.item())
    
    with open(config.log_fname, 'a') as f:
        f.write(message)


# Write time to file
def log_time(config, epoch, start_time):
    message = f'\nTrain time for epoch {epoch}: {datetime.datetime.now()-start_time}\n' 

    with open(config.log_fname, 'a') as f:
        f.write(message)


# Clears file
def newfile(config):
    with open(config.log_fname, 'w') as _:
        pass

    with open(config.log_extra_fname, 'w') as _:
        pass


# Logs stuffs
def log_extra(config, rank, epoch, batch, info):
    acc_real, acc_fake, preds_real, preds_fake = info
    message1 = f'''
    Rank {rank} | epoch {epoch} | batch {batch} 
    Acc. real: {acc_real} | Preds for real {preds_real.squeeze()}: {preds_real.squeeze()[:10]}
    Acc. fake: {acc_fake} | Preds for fake {preds_fake.squeeze().size()}: {preds_fake.squeeze()[:10]}
    '''

    with open(config.log_extra_fname, 'a') as f:
        f.write(message1)