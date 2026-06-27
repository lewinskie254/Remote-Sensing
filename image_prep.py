import glob 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from patchify import patchify, unpatchify
import tifffile
import tqdm 

from tqdm import tqdm  # Use from tqdm.notebook import tqdm if in Jupyter

def fetch_images(path, color=True, extension='png'):
    images = []
    all_img_paths = []
    
    ext = extension.replace('*', '').replace('.', '').lower()
    
    for directory_path in glob.glob(path):
        if not os.path.isdir(directory_path):
            continue
            
        if ext in ['png', 'jpg', 'jpeg']:
            for e in [ext, ext.upper()]:
                all_img_paths.extend(glob.glob(os.path.join(directory_path, f"*.{e}")))
        elif ext in ['tiff', 'tif']:
            for e in ['tif', 'TIF', 'tiff', 'TIFF']:
                all_img_paths.extend(glob.glob(os.path.join(directory_path, f"*.{e}")))

    if not all_img_paths:
        print(f"No files found matching extension '{extension}' in the provided path.")
        return np.array([])

    for img_path in tqdm(all_img_paths, desc="Loading Images"):
        
        if ext in ['png', 'jpg', 'jpeg']:
            flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(img_path, flag)
            if img is None:
                print(f"Failed to load standard image: {img_path}")
                continue

        elif ext in ['tiff', 'tif']:
            img = tifffile.imread(img_path)
            if img is None:
                print(f"Failed to load TIFF image: {img_path}")
                continue
            
            if not color and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        images.append(img)
    
    try:
        images = np.array(images)
    except ValueError as e:
        print("\n[WARNING] Array creation failed. Images have mismatched dimensions.")
        print(f"Error details: {e}\n")

    return images


def get_patches(image, mask, dims = (256, 256), step=256):
    print("Image Shape", image.shape)
    print("Mask Shape", mask.shape)

    
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

def save_patches(patches, format='png', path='patches/images'): 
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            if format.lower() in ['.png', '.jpg', '.jpeg']:
                cv2.imwrite(path + 'patch_' + str(i) + str(j) + '.' + format.lower(), single_patch)
            elif format.lower() in ['tiff', '*.tif']: 
                tifffile.imwrite(path + 'patch_'+ str(i) + str(j) + '.' + format.lower(), single_patch)



