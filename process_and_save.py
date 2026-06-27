#%%
import segmentation_models_pytorch as sm 
from image_prep import fetch_images, save_patches_and_masks, get_patches 
from tqdm import tqdm



#%%
#Import Image sizes (Will be varied for experiments )
# to see how it performs on different sized models 
SIZE_X = 256
SIZE_Y = 256 


#%%
train_images = fetch_images('png/train')

#%%
train_masks = fetch_images('png/train_labels', False)

#%%
test_images = fetch_images('png/test')
test_masks = fetch_images('png/test_labels', False)

#%%
train_patches = [ get_patches(train_images[i], train_masks[i], (SIZE_X, SIZE_Y), SIZE_X) for i in tqdm(range(len(train_images)))] #default size 256 
test_patches = [ get_patches(test_images[i], test_masks[i], (SIZE_X, SIZE_Y), SIZE_X) for i in tqdm(range(len(test_images)))] #default size 256 

#%%
print(train_patches[0][0].shape)



#%%
save_patches_and_masks(train_patches, image_dir='patches/train', mask_dir='patches/train_labels')
#%%
save_patches_and_masks(test_patches, image_dir='patches/test', mask_dir='patches/test_labels')

# %%
