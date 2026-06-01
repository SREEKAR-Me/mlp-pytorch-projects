import numpy as np
import pandas as pd
import torch

import torch.nn as nn
class HousingMLP(nn.Module):
    def __init__(self,input_dim=8,hidden_dims=[264,128,64],dropout=0.3):
        super().__init__()

        layers=[]
        prev_dim=input_dim

        for hid in hidden_dims:
            layers+=[
                nn.Linear(prev_dim,hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim=hid

        layers.append(nn.Linear(prev_dim,1))

        self.mlp=nn.Sequential(*layers)

    def forward(self,X):
        return self.mlp(X)
