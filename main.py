import glob 
import segmentation_models_pytorch as sm 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from patchify import patchify, unpatchify
import tifffile
from .image_prep import fetch_images, save_patches, get_patches 




#Import Image sizes (Will be varied for experiments )
# to see how it performs on different sized models 
SIZE_X = 256
SIZE_Y = 256 




BACKBONE = 'resnet34'
process_input = sm.get_preprocessing(BACKBONE)

train_images = fetch_images('png/train')
train_masks = fetch_images('png/train_masks', False)

test_images = fetch_images('png/test')
test_masks = fetch_images('png/test_labels', False)
train_patches = [ get_patches(train_images[i], train_masks[i], (SIZE_X, SIZE_Y), SIZE_X) for i in range(len(train_images))] #default size 256 
test_patches = [ get_patches(test_images[i], test_masks[i], (SIZE_X, SIZE_Y), SIZE_X) for i in range(len(test_images))] #default size 256 