import torch
from torch.utils.data import Dataset,DataLoader
import sklearn
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data=pd.read_csv("creditcard.csv")

class FraudDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    

def creditloader(batch_size=512):
    df=pd.read_csv("creditcard.csv")
    X=df.drop(["Class","Time"],axis=1)
    y=df["Class"].values

    X_train,X_t,y_train,y_t=train_test_split(X,y,test_size=0.3,random_state=42)
    X_test,X_val,y_test,y_val=train_test_split(X_t,y_t,test_size=0.5,random_state=42)

    scaler=StandardScaler()
    X_train['Amount']=scaler.fit_transform(X_train[['Amount']])
    X_test['Amount']=scaler.fit_transform(X_test[['Amount']])
    X_val['Amount']=scaler.fit_transform(X_val[['Amount']])

    X_train = X_train.values
    X_val   = X_val.values
    X_test  = X_test.values

    n_zero=(y_train==0).sum()
    n_one=(y_train==1).sum()

    weight_diff=torch.tensor([n_one/n_zero])

    X_train_ds=FraudDataset(X_train,y_train)
    X_test_ds=FraudDataset(X_test,y_test)
    X_val_ds=FraudDataset(X_val,y_val)

    return (
        DataLoader(X_train_ds,batch_size=batch_size,shuffle=True),
        DataLoader(X_test_ds,batch_size=batch_size),
        DataLoader(X_val_ds,batch_size=batch_size),
        weight_diff
    )