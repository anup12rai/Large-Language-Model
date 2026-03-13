import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C//self.heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        attn = (q @ k.transpose(-2,-1)) / (C**0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B,N,C)
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.net(x)