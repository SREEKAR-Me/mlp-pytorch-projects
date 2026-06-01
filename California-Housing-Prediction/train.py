import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import housingloader
from model import HousingMLP
from torch.optim import Adam

def train(epoch,learn_r):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader,test_loader,val_loader=housingloader()
    model=HousingMLP().to(device)
    criterion=nn.MSELoss()
    optimizer=Adam(model.parameters(),lr=learn_r,weight_decay=1e-4)

    best_val_loss=99999
    for epoch in range(epoch):
        model.train()
        train_loss=0
        for X,y in train_loader:
            X_batch,y_batch=X.to(device),y.to(device)

            optimizer.zero_grad()
            preds=model(X_batch)
            loss=criterion(preds,y_batch)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        model.eval()
        val_loss=0
        with torch.no_grad():
            for X_val,y_val in val_loader:
                X_val_batch=X_val.to(device)
                y_val_batch=y_val.to(device)

                preds=model(X_val_batch)
                val_loss+=criterion(preds,y_val_batch)

        train_loss/= len(train_loader)
        val_loss/= len(val_loader)
        if val_loss <best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(),"best_model.pt")
        
        print("Validation Loss", val_loss, "Training Loss: ",train_loss, "Epoch: ", epoch)
        
    print("Training complete. Best valuation loss ",best_val_loss)


if __name__ =="__main__":
    train(100,1e-3)

