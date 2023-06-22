import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten
from .nets_utils import EmbeddingRecorder
import torch

class ConvNN(nn.Module):
    """
    Simple CNN for CIFAR10
    """
    
    def __init__(
            self, 
            channel, 
            num_classes, 
            im_size, 
            record_embedding: bool = False , 
            no_grad: bool = False, 
            pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        
        super(ConvNN, self).__init__()
        self.conv_32 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv_64 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv_96 = nn.Conv2d(64, 96, kernel_size=3, padding='same')
        self.conv_128 = nn.Conv2d(96, 128, kernel_size=3, padding='same')
        self.fc_512 = nn.Linear(512, 512)
        self.fc_10 = nn.Linear(512, 10)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_10
    
    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # out = x.view(x.size(0), -1)
            out = x
            out = self.conv_32(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = self.conv_64(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = self.conv_96(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = self.conv_128(out)
            out = self.relu(out)
            out = self.max_pool(out)

            out = self.flatten(out)

            out = self.embedding_recorder(out)

            out = self.fc_512(out)
            out = self.relu(out)
            out = self.fc_10(out)

        return out