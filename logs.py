import torch
from tqdm import tqdm


# Print num parameters, num GPUs
def print_pytorch_stats(G, D):
    print('Num trainable Discriminator:', sum(p.numel() for p in D.parameters()))
    print('Num trainable Generator:', sum(p.numel() for p in G.parameters()))
    print('Using {} GPUs'.format(torch.cuda.device_count()))


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


# Print hyperparameters
def print_hyperparameters(num_epochs, batch_size, nchannels, img_size, latent_size, update_every):
    print('Num epochs {}, batch size {}'.format(num_epochs, batch_size))
    print('Updating discriminator every {}'.format(update_every))
    print('Images are ({}, {}, {}) latent size {}'.format(nchannels, img_size, img_size, latent_size))