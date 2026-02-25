import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class FlowCNNEncoder(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # Input: (B, 2, H, W) -> u, v channels
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # 특징 응축을 위한 Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x: (B, 2, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        
        x = self.avgpool(x) # (B, 512, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x) # (B, 1024)
        return x

class FlowViTEncoder(nn.Module):
    def __init__(self, img_size=(192, 320), patch_size=16, embed_dim=1024):
        super().__init__()
        h, w = img_size
        num_patches = (h // patch_size) * (w // patch_size)
        patch_dim = 2 * patch_size * patch_size # 2 channels (u, v)

        # 1. Flow 이미지를 패치로 나누고 평탄화(Flatten)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 2. 위치 정보 추가 (Learned Positional Embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # x: (B, 2, H, W)
        x = self.to_patch_embedding(x) # (B, num_patches, embed_dim)
        x += self.pos_embedding
        
        # 이 토큰들을 그대로 Transformer에 넣거나, 평균내어 하나의 벡터로 사용
        return x # (B, num_patches, 1024)