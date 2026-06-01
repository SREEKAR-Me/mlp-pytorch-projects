import torch
import torch.nn as nn


class CreditCardMLP(nn.Module):
    def __init__(self,input_dim=29,hidden_dim=[256,128,64],dropout=0.4):
        super().__init__()

        prev_dim=input_dim
        layers=[]

        for hid in hidden_dim:
            layers+=[
                nn.Linear(prev_dim,hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim=hid

        layers.append(nn.Linear(prev_dim,1))
        self.net=nn.Sequential(*layers)

    def forward(self,X):
        return self.net(X)



