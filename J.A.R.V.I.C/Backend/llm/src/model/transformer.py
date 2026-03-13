import torch
import torch.nn as nn
from .layers import SelfAttention, FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_hidden):
        super().__init__()
        self.attn = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_hidden=128, num_layers=2, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_hidden) for _ in range(num_layers)])
    
    def forward(self, x):
        B, N = x.shape
        positions = torch.arange(0,N).unsqueeze(0).expand(B,N).to(x.device)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)  # Mean pooling