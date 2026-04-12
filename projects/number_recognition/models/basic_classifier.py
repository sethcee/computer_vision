import torch 
import torch.nn as nn

class BasicClassifier(nn.Module) :

    def __init__(self, num_classes) :
        super().__init__()

        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x) :
        x = self.flatten(x)
        x = self.layers(x)
        return x