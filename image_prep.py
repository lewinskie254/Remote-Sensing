import glob 
import segmentation_models_pytorch as sm 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 


BACKBONE = 'resnet34'
process_input = sm.get_preprocessing(BACKBONE)


#Import Image sizes (Will be varied for experiments )
# to see how it performs on different sized models 
SIZE_X = 256
SIZE_Y = 256 

def fetch_images(path, color=True):
    images = []
    for directory_path in glob.glob(path):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            
            #lets get the image tensor 
            element = cv2.IMREAD_COLOR if color else 0 

            img = cv2.imread(img_path, element)

            images.append(img)
    
    # to prep the images for machine learning processing ...
    # lets make them numpy arrays 
    images = np.array(images)

    return images 


train_images = fetch_images('train/images')
train_masks = fetch_images('mask_path', False)