import os
import cv2
import torch 
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from PIL import Image
from glob import glob
import torch.optim as optim
from natsort import natsorted
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch_lr_finder import LRFinder

from network import ViT

class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, mode='train', transform=None):
        self.data_path = data_path 
        self.transform = transform
        self.mode = mode
        self.label_path = label_path

        self.data = pd.read_csv(self.label_path)
        self.image_paths = self.data["image:FILE"].tolist()

        raw_labels = self.data["category"].tolist()
        self.label_map = {name:idx for idx, name in enumerate(sorted(set(raw_labels)))}
        self.labels = [self.label_map[l] for l in raw_labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_paths[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss=0
    total_correct=0
    total_samples=0

    all_preds=[]
    all_labels=[]

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # By Multiplying batch_size (imgs.size(0)), Get the total sum of losses.
            total_loss += loss.item() * imgs.size(0) 
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, all_preds, all_labels



def train(model, train_loader, val_loader, device, epochs=20, lr=3e-4):
    wandb.init(
        project="vit-plant-30class",
        config={
            "epochs": epochs,
            "lr": lr,
            "batch_size": train_loader.batch_size,
            "architecture": "ViT"
        }
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=val_preds,
                y_true=val_labels
            )
        })

        print(f"[{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    wandb.finish()



def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc, preds, labels



### Heat map Visualization ###

'''
시각화 아이디어 
1. Patch 별로 쪼개진 것을 PE 를 통해 다시 모으기
2. Attention Score에 따른 HeatMap 생성

'''
def vis_heatmap():
    pass


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    HEAD_PATH = "./CVIP/ViT/dataset"

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_dataset = CustomDataset(
        HEAD_PATH,
        os.path.join(HEAD_PATH, "train.csv"),
        "train",
        transform_train
    )

    val_dataset = CustomDataset(
        HEAD_PATH,
        os.path.join(HEAD_PATH, "val.csv"),
        "val",
        transform_eval
    )

    test_dataset = CustomDataset(
        HEAD_PATH,
        os.path.join(HEAD_PATH, "test.csv"),
        "test",
        transform_eval
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ViT(num_classes=30).to(device)

    print(f"Using device: {device}")
    
    print("LR Finder starts..")

    temp_criterion = nn.CrossEntropyLoss()
    temp_optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.05)

    lr_finder = LRFinder(model, temp_optimizer, temp_criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=200, step_mode='exp')

    loss_history = lr_finder.history['loss']
    lr_history = lr_finder.history['lr']

    min_grad_idx = (np.gradient(np.array(loss_history))).argmin()

    best_lr = lr_history[min_grad_idx] / 10.0
    print(f"Best Learning Rate : {best_lr:.6f}")
    lr_finder.reset()
    
    
    print("Starting Training...")
    train(model, train_loader, val_loader, device, lr=best_lr)

    
    print("Loading Best Model...")
    model.load_state_dict(torch.load("best_model.pth"))

    
    print("Starting Testing...")
    test(model, test_loader, device)


if __name__ == "__main__":
    main()