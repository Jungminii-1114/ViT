import math
import torch
import torch.nn as nn

'''
Multi-Head Self-Attention + MLP + Residual + Layer Normalization
'''

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.um_patches = (self.img_size // self.patch_size) ** 2

        self.LinProjection = nn.Linear(
            in_channels * self.patch_size * self.patch_size, # P^2 x C
            emb_size
        )
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        P = self.patch_size

        x = self.unfold(x)
        x = x.permute(0, 2, 1)
        x = self.LinProjection(x)

        return x
    
'''
Class name 'PatchEmbedding' defined right above actually the same as Conv2d.

Conv2d is same with adapting a single Linear Layer to each patch.

We can write it down like below.
'''

class PatchEmb_with_conv(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.img_size=img_size
        self.patch_size = patch_size
        self.n_patches = (self.img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2) # As we only have to flatten the spatial dim, use flatten(2) : (B, D, HW)
        x = x.transpose(1, 2)

        return x

class Attention(nn.Module):
    def __init__(self, emb_size=768, n_heads=12):
        super().__init__()
        self.emb_size=emb_size
        self.num_heads = n_heads
        self.head_dim = emb_size // self.num_heads

        self.V = nn.Linear(emb_size, emb_size)
        self.K = nn.Linear(emb_size, emb_size)
        self.Q = nn.Linear(emb_size, emb_size)

        self.out_lin = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        B, L, D = x.size()

        V = self.V(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.K(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q = self.Q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        AttentionScore = torch.matmul(Q, K.transpose(-2, -1))
        AttentionScore /= math.sqrt(self.head_dim)
        AttentionScore = torch.softmax(AttentionScore, dim=-1) # In order to make weighted sum : 1
        result=torch.matmul(AttentionScore, V)
        
        # head concat
        result = result.transpose(1, 2).contiguous().view(B, L, D)
        result = self.out_lin(result)

        return result

class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        # 이거 이름 바뀌어야 하지 않나?
        hidden_dim = emb_size * 4

        self.fc1 = nn.Linear(emb_size, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # (B, L, D)
        x = self.fc1(x)  # (B, L, in_features) -> (B, L, hidden_features)
        x = self.gelu(x)

        x = self.fc2(x)  # (B, L, hidden) -> (B, L, out_features)
        x = self.dropout(x)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_size, n_heads):
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(emb_size)
        self.attention = Attention(emb_size, n_heads)
        self.layerNorm2 = nn.LayerNorm(emb_size)
        self.mlp = MLP(emb_size)

    def forward(self, x):
        x = x + self.attention(self.layerNorm1(x))

        x = x + self.mlp(self.layerNorm2(x))

        return x
    
class EncoderStacked(nn.Module):
    def __init__(self, depth, emb_size, n_heads):
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock(emb_size, n_heads) for _ in range(depth)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, num_classes, embed_dim=768, depth=3, n_heads=3, img_size=224, patch_size=16, in_channels=3):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmb_with_conv(img_size=img_size, in_channels=in_channels, patch_size=patch_size, emb_size=embed_dim)
        num_patches = self.patch_embed.n_patches

        self.posEmb = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.CLSToken = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.encoder = EncoderStacked(depth=depth, emb_size=embed_dim, n_heads=n_heads)

        # Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x) # (B, N, D)
        cls_tokens = self.CLSToken.expand(B, -1, -1) #(B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)

        x = x + self.posEmb
        x = self.encoder(x) #(B, N+1, D)
        cls_token_final = x[:, 0] #(B, D) : as we have to put this into head

        result = self.head(cls_token_final)

        return result




        