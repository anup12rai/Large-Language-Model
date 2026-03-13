import torch.nn as nn

class QueryClassifier(nn.Module):
    def __init__(self, transformer, num_classes=3):
        super().__init__()
        self.transformer = transformer
        self.fc = nn.Linear(transformer.embed.embedding_dim, num_classes)
    
    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)