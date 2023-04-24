import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torch
import os
import random

# Load and preprocess data here
def convert_file_paths(dataset_type, class_label):
    '''
    Converts file paths of the form './datasets/<train or test>/<cat or dog>.xxx.jpg' into
    './datasets/<train or test>/<real or fake>/<cat or dog>.xxx.jpg' so that they can be 
    parsed by the Pytorch DataLoader object. 
    Can be revised and used for adding more data in the future (such as pictures of birds)
    The folders titled 'real' and 'fake' define the class assignments.

    :params: 
    dataset_type: string of 'train' or 'test'. Title of the folder whose images are to be further sorted 
    class_label: string of 'real' or 'fake'. Title of the folder to which the images are moved
    
    :returns: 
    None
    '''
    # Convert path names from cat.N.jpg to ./cat/N.jpg
    directory = f'./datasets/{dataset_type}'
    char_num = len(directory)
    # iterate over files in that directory
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # Checking if it is a file. Will exclude created or existing folders
        if os.path.isfile(file):
            new_path = f'./datasets/{dataset_type}/{class_label}/{file[char_num:]}'
            os.rename(file, new_path)
        else:
            pass

    return



def load_data(dataset_type, class_label):
    '''
    Loads data from a specified directory and reformats it. Each element in the returned list is a tuple,
    (img, label)

    :params:
    dataset_type: string of 'train' or 'test'. Title of the folder from which to load data 
    class_label: string of 'real' or 'fake'. Title of the folder from which to load data
    
    :returns: 
    data: a python list, where each element is a tuple of an image(np.ndarray of shape (3, 64, 64)) 
          and its label (np.ndarray of shape (1) with element 0 or 1. 0 --> real; 1 --> fake)
    '''
    data_set = f'./datasets/{dataset_type}/{class_label}'
    transform = transforms.Compose([transforms.Resize(64),
                                 transforms.CenterCrop(64),
                                 transforms.ToTensor()])
    training_dataset = datasets.ImageFolder(data_set, transform=transform)
    data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=16, shuffle=True)

    
    # edit so that the batch_size parameter can be changed and multiple pictures can be loaded at once

    
    data = []
    # Populate data array with images and their labels
    for img, label in data_loader:
        img = np.squeeze(img)
        label = np.asarray(label)
        # Check and convert BW --> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        # Add image and label as tuple to the data list
        data.append((img, label))

    return data



def shuffle_real_and_fake(real_imgs, fake_imgs):
    '''
    Creates a single, randomly ordered list of image and label tuples from two lists of 
    real and fake images, respectively.

    :params:
    real_imgs: a list of length n composed of tuples of real images from the data set and their labels: (img, label)
    fake_imgs: a list of length m composed of tuples of real images from the data set and their labels: (img, label)

    :returns:
    img_shuffle: an np.ndarray of shape [n+m, 3, 64, 64] of randomly ordered real and fake images
    label_shuffle: an np.ndarray of shape [n+m] of the corresponding label for images in random_img_shuffle

    '''
    # Concatenate lists
    composite = real_imgs + fake_imgs
    reordered = random.shuffle(composite)
    # Unpack tuples
    img_shuffle = np.array([reordered[i][0] for i in range(len(composite))])
    label_shuffle = np.array([reordered[i][1] for i in range(len(composite))])
    
    return img_shuffle, label_shuffle



def standardize(imgs):
    '''
    Standarizes a sets of images in place before they are read by the descriminator.

    :params:
    imgs: an np.ndarray of shape [num_imgs, 3, img_size, img_size]

    :returns:
    None
    '''
    mean = np.sum(imgs, axis=0)/len(imgs)
    std = np.std(imgs, axis=0)
    imgs = (imgs-mean)/std 
    return


def create_generator_seeds(batch_size):
    '''
    Creates a new set of random seed vectors for the generator

    :params:
    batch_size: the number of seeds to generate for each training iteration

    :returns:
    generator_seeds: an np.ndarray of shape [batch_size, 1, 100] with random entries on [0,1]
    '''

    generator_seeds = np.random.rand(batch_size, 1, 100)

    return generator_seeds
