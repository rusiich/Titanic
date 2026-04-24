import torch.nn as nn
from configs import conf

class DNN(nn.Module):
    def __init__(self, input, output=conf.general.num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 256),
            nn.BatchNorm1d(256),
            nn.Relu(),
            
            nn.Dropout(0.2),
            nn.nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Relu(),

            nn.Dropout(0.2),
            nn.nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Relu(),

            nn.Dropout(0.2),
            nn.nn.Linear(128, output),
        ).to(conf.trtaining.device)


    def forward(self, x):
        return self.model(x)