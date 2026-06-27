
#%%
import segmentation_models_pytorch as sm
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from image_prep import fetch_images
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from data_loader import SegmentationDataset


load_dotenv()
TOKEN = os.environ['HF_TOKEN']

login(token=TOKEN)


#%%
BACKBONE = 'resnet34'

preprocess_input = get_preprocessing_fn(BACKBONE)

model = sm.Unet(
    encoder_name=BACKBONE, 
    encoder_weights='imagenet', 
    in_channels=3, 
    classes=1,
)

#%%
X_train_path = r'patches\train'
y_train_path = r'patches\test'

#%%
X_train = fetch_images(X_train_path)
y_train = fetch_images(y_train_path, False)



# %%
from torch.utils.data import DataLoader

dataset = SegmentationDataset(X_train, y_train, preprocess_input)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

