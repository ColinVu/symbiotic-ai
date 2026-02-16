"""Neural network classifier for CLIP embeddings."""

import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Lightweight classifier head for CLIP embeddings.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear
    
    This is intentionally simple - the heavy lifting is done by CLIP.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


__all__ = ['ClassifierHead']
