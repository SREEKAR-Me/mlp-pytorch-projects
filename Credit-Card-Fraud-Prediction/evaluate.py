import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from model import CreditCardMLP
from dataset import creditloader

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,test_loader,_,_=creditloader()

    model=CreditCardMLP().to(device)
    model.load_state_dict(torch.load("best_model.pt",weights_only=True))
    model.eval()

    pred=[]
    target=[]
    with torch.no_grad():
        for X,y in test_loader:

            preds=model(X.to(device)).cpu().numpy()
            pred.append(preds)
            target.append(y.numpy())

    pred=np.concatenate(pred)
    target=np.concatenate(target)
    roc_score = roc_auc_score(target, pred)
    print(f"Test ROC-AUC score: {roc_score}")
    

if __name__ == "__main__":
    evaluate()   