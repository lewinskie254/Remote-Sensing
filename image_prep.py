import glob 
import segmentation_models_pytorch as sm 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from patchify import patchify, unpatchify
import tifffile


BACKBONE = 'resnet34'
process_input = sm.get_preprocessing(BACKBONE)


#Import Image sizes (Will be varied for experiments )
# to see how it performs on different sized models 
SIZE_X = 256
SIZE_Y = 256 

def fetch_images(path, color=True, extension='*.png'):
    images = []
    for directory_path in glob.glob(path):
        for img_path in glob.glob(os.path.join(directory_path, extension)):
            
            if extension.lower() in ['*.png', '*.jpg', '*.jpeg']:
                # OpenCV logic
                flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
                img = cv2.imread(img_path, flag)

            elif extension.lower() in ['*.tiff', '*.tif']:
                # Tifffile logic
                img = tifffile.imread(img_path)
                
                # Manually convert TIFF to grayscale if requested and it has 3 channels
                if not color and len(img.shape) == 3:
                    # Assumes RGB/BGR channel structure
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            images.append(img)
    
    # to prep the images for machine learning processing ...
    # lets make them numpy arrays 
    try:
        images = np.array(images)
    except ValueError as e:
        print("\n[WARNING] Could not convert to a single NumPy array!")
        print("Your images likely have mismatched dimensions or channel counts.")
        print(f"Error detail: {e}\n")

    return images 



def extract_tiles(image, mask, tile_size=256):

    image_tiles = []
    mask_tiles = []

    H, W = mask.shape

    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):

            image_tile = image[
                y:y+tile_size,
                x:x+tile_size
            ]

            mask_tile = mask[
                y:y+tile_size,
                x:x+tile_size
            ]

            image_tiles.append(image_tile)
            mask_tiles.append(mask_tile)

    return image_tiles, mask_tiles

train_images = fetch_images('train/images')
train_masks = fetch_images('mask_path', False)