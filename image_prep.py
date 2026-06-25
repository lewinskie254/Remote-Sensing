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


def get_patches(image, mask, dims = (256, 256), step=256):
    if not isinstance(dims, tuple):
        print("dims must be a 1 by 2 tuple eg. (256, 256)")
        return 
    # Ensure inputs are numpy arrays
    image, mask = np.array(image), np.array(mask)
    
    # Check if dimensions are divisible by the step/patch size
    if (image.shape[0] % step != 0) or (image.shape[1] % step != 0):
        print("Warning: Image dimensions are not divisible by patch size. Patchify may fail.")
        
    try:
        patches_image = patchify(image, (*dims, image.shape[-1]), step=step)
        patches_mask = patchify(mask, (*dims, 1), step=step)
        return patches_image, patches_mask
    except Exception as e:
        print(f"Error during patch extraction: {e}")
        return None, None


train_images = fetch_images('train/images')
train_masks = fetch_images('mask_path', False)