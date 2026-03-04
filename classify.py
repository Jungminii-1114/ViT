import os
import cv2
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from glob import glob
from torch import Tensor
import torch.optim as optim
from natsort import natsorted
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

from network import ViT

HEAD_PATH = "./CVIP/ViT/dataset"

train_path = os.path.join(HEAD_PATH, "train")
val_path = os.path.join(HEAD_PATH, "val")
test_path = os.path.join(HEAD_PATH, "test")

train_label = os.path.join(HEAD_PATH, "train.csv")
val_label = os.path.join(HEAD_PATH, "val.csv")
test_label = os.path.join(HEAD_PATH, "test.csv")

num_train_img = len(os.listdir(train_path)) 
''' 31 : Folders + .txt (class num : 30) '''

train_folders = sorted([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
''' train_folders : "aloevera", "banana", "bilbi", ...'''

aloe_folder_path = os.path.join(train_path, train_folders[0])
aloe_imgs = natsorted(os.listdir(aloe_folder_path))
first_img_path = os.path.join(aloe_folder_path, aloe_imgs[0])


# Aloe Image Visualization Code !! 
aloe_first_img = cv2.imread(first_img_path)
print(aloe_first_img.shape) # (640, 960, 3)

'''
cv2.imshow("Aloe first", aloe_first_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

breakpoint()

# Transforms

'''
패치 추출 라이브러리 : unfold() 혹은 rearrange 이용해보기 
Augmentation -> ToTensor -> Normalize
LRFinder 이용해서 최적의 학습률 찾아보기 
'''

changed_img = Image.open(first_img_path)
print(changed_img) # 960, 640, RGB 


transform_opt1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_opt2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensored_img = transform_opt1(changed_img)
tensor_img = tensored_img.unsqueeze(0)
print("="* 30)
print(f"After transform Before unfold : {tensor_img.shape}") # ([1, 3, 224, 224])

unfold = nn.Unfold(kernel_size=(16, 16), stride=16 ,padding=0) 
x = unfold(tensor_img)
print(x.shape) # ([1. 768, 196]) - 768 : 하나의 패치를 Flatten한 벡터 길이  (임베딩)  196 : 패치 개수 
print("="*30)



# Criterion and Optimizer

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(num_classes=30).to(device)
criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=3e-4)

