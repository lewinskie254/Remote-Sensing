#%%
import segmentation_models_pytorch as sm 
from image_prep import fetch_images, save_patches, get_patches 
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
train_images[0].shape

#%%
train_masks[0].shape


#%%
print(train_patches)
#%%
save_patches(train_patches, 'png', 'patches/train')
save_patches(test_patches, 'png', 'patches/test')


#%%
BACKBONE = 'resnet34'
process_input = sm.get_preprocessing(BACKBONE)