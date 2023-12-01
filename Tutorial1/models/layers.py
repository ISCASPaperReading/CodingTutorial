import torch.nn as nn

def create_layers(in_dim,hidden_dim,out_dim):
    layer = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )
    return layer