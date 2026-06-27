import glob 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from patchify import patchify, unpatchify
import tifffile
from pathlib import Path
from tqdm import tqdm  

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


def get_patches(image, mask, dims=(256, 256), step=256):
    # 1. Transform DIMs (1500, 1500) -> (1500, 1500, 1)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
        
    # 2. Calculate padding
    h, w = image.shape[0], image.shape[1]
    pad_h = (step - (h % step)) % step
    pad_w = (step - (w % step)) % step
    
    # 3. Apply padding
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    # 4. Patchify
    patches_image = patchify(image_padded, (*dims, image.shape[-1]), step=step)
    patches_mask = patchify(mask_padded, (*dims, 1), step=step)
    
    return patches_image, patches_mask

def save_patches_and_masks(patch_list, image_dir='patches/images', mask_dir='patches/masks'):
    # Create Train and Label path directories
    img_path = Path(image_dir)
    mask_path = Path(mask_dir)
    img_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    
    for img_idx, (img_patches, mask_patches) in enumerate(tqdm(patch_list, desc="Saving patches")):
        #Flatten image (6, 6, 1, 256, 256, 3) -> (36, 256, 256, 3)
        flat_img = img_patches[0, 0].reshape(-1, *img_patches.shape[3:])
        
        #Flatten mask (6, 6, 1, 256, 256, 1) -> (36, 256, 256, 1)
        flat_mask = mask_patches[0, 0].reshape(-1, *mask_patches.shape[3:])
        

        for patch_idx, (img_patch, mask_patch) in enumerate(zip(flat_img, flat_mask)):
            # filenames
            img_filename = img_path / f"img{img_idx:03d}_patch{patch_idx:03d}.png"
            mask_filename = mask_path / f"img{img_idx:03d}_patch{patch_idx:03d}.png"
            
            # Save as UINT 8 - normal for pics 
            img_save = img_patch.astype(np.uint8)
            mask_save = mask_patch.astype(np.uint8)
            
            # Save files
            cv2.imwrite(str(img_filename), img_save)
            cv2.imwrite(str(mask_filename), mask_save)
            
    print("Successfully saved all image and mask patches.")