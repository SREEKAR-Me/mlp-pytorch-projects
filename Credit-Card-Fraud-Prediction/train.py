import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from model import CreditCardMLP
from dataset import creditloader
from sklearn.metrics import roc_auc_score

def train():
    device=torch.device("cuda")
    tr_load,te_load,val_load,weigh_diff=creditloader()
    model=CreditCardMLP().to(device)
    optimizer=Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    criterion=nn.BCEWithLogitsLoss(pos_weight=weigh_diff.to(device))

    best_score=0
    for epoch in range(50):
        model.train()
        for X_batch,y_batch in tr_load:
            X=X_batch.to(device)
            y=y_batch.to(device)

            optimizer.zero_grad()
            pred=model(X)
            loss=criterion(pred,y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_pred=[]
        total_target=[]
        with torch.no_grad():
            for X_batch,y_batch in val_load:
                logits = model(X_batch.to(device)).squeeze().cpu()
                probs  = torch.sigmoid(logits)
                total_pred.append(probs.numpy())
                total_target.append(y_batch.squeeze().numpy())
            
        auc_score=roc_auc_score(np.concatenate(total_target),np.concatenate(total_pred))

        if auc_score>best_score:
            best_score=auc_score
            torch.save(model.state_dict(),"best_model.pt")

        print("AUC Score: ",auc_score, "Epoch: ", epoch)
    
    print("Best AUC score: ", best_score)


if __name__ =="__main__":
    train()