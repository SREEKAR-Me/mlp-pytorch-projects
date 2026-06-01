import torch
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from model import HousingMLP
from dataset import housingloader

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,_,test_loader=housingloader()

    model=HousingMLP().to(device)
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
    rmse = np.sqrt(mean_squared_error(target, pred))
    r2   = r2_score(target, pred)
    print(f"Test RMSE: {rmse:.4f} | R²: {r2:.4f}")
    

if __name__ == "__main__":
    evaluate()   