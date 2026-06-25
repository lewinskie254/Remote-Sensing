from PIL import Image
import numpy as np

img = np.array(Image.open(
    "png/train/22678915_15.png"
))

mask = np.array(Image.open(
    "png/train_labels/22678915_15.png"
))

print("Image shape:", img.shape)
print("Image dtype:", img.dtype)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)

print("Mask unique values:")
print(np.unique(mask))