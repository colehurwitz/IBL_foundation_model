# This file contains the implementation of the r2 score metric
from torcheval.metrics import R2Score
def r2_score(y_true, y_pred, device="cpu"):
    metric = R2Score()
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    metric.update(y_pred, y_true)
    return metric.compute().item()