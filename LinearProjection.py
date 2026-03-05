import torch 
import timm
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# Linear Projection 

# Linear Projection (전)
# (B, N, P^2*C)

# Linear Projection (후)
# (B, N, D)


import torch
import matplotlib.pyplot as plt

# 가상의 pre-trained ViT 모델 가중치 불러오기 (예: timm 라이브러리 활용)
# 가중치 형태가 (D, C, P, P) -> (768, 3, 16, 16) 이라고 가정
model = timm.create_model('vit_base_patch16_224', pretrained=True)
weights = model.patch_embed.proj.weight.data.cpu()

# 시각화를 위해 가중치 값의 범위를 0~1 사이로 정규화 (Min-Max Scaling)
weights = (weights - weights.min()) / (weights.max() - weights.min())

# 768개의 필터 중 상위 16개만 그려보기
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    # PyTorch의 (C, P, P) 순서를 Matplotlib 시각화를 위해 (P, P, C)로 변경
    img = weights[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis('off')
    
plt.tight_layout()
plt.show()