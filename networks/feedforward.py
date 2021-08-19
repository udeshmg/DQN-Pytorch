import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class FeedForward(nn.Module):

    def __init__(self, num_layers, embed_dim, input_dim, outputs, device):
        super(FeedForward, self).__init__()

        self.device = device

        # Create Network
        layers = []
        layers.append(nn.Linear(input_dim, embed_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(embed_dim, outputs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x #self.head(x.view(x.size(0), -1))