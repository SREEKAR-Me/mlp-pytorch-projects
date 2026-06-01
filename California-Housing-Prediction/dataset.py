import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=fetch_california_housing()
data
class HousingDataset(Dataset):

    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
def housingloader(batch_size=32):
    data=fetch_california_housing()
    X=data.data
    y=data.target
    X_train,X_t,y_train,y_t=train_test_split(X,y,test_size=0.3, random_state=42)
    X_test,X_val,y_test,y_val=train_test_split(X_t,y_t,test_size=0.5,random_state=42)

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)
    X_val=sc.fit_transform(X_val)

    X_train_ds=HousingDataset(X_train,y_train)
    X_test_ds=HousingDataset(X_test,y_test)
    X_val_ds=HousingDataset(X_val,y_val)

    return (DataLoader(X_train_ds,batch_size=batch_size,shuffle=True),
            DataLoader(X_test_ds,batch_size=batch_size),
            DataLoader(X_val_ds,batch_size=batch_size))

