
#%%
import segmentation_models_pytorch as sm
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from image_prep import fetch_images
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from segmentation_loader import SegmentationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from partial_cross_entropy import PartialBCELoss 



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
dataset = SegmentationDataset(X_train, y_train, preprocess_input)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#%% 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = PartialBCELoss() 

epochs = 20
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(loader):.4f}")

# Save the model after training
torch.save(model.state_dict(), "unet_resnet34_model.pth")