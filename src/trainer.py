import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model import UNet
from dataset import SpineDataset

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        predict = predict.view(-1)
        target = target.view(-1)
        intersection = (predict * target).sum()
        return 1 - ((2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth))

def train_model(train_dir, val_dir, output_model_path, epochs=10, lr=1e-4, batch_size=2, device="cuda"):
    dataset_train = SpineDataset(train_dir)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceLoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            for images, masks in loader_train:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix(**{"loss": loss.item()})
        
        logger.info(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(loader_train)}")
        
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Model saved to {output_model_path}")
