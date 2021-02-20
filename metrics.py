import torch
import numpy as np 

def ADE(t_true, t_pred, batch_mean=True):
    t_true = t_true.permute(2, 1, 0)[:2].clone()
    t_pred = t_pred.permute(2, 1, 0)[:2].clone()

    xn_true = t_true[0]
    xn_pred = t_pred[0]
    yn_true = t_true[1]
    yn_pred = t_pred[1]

    x_diff = xn_true-xn_pred
    y_diff = yn_true-yn_pred

    sqroot = torch.sqrt(x_diff * x_diff + y_diff * y_diff)
    FDE_result = sqroot.cpu().detach().numpy()

    if batch_mean:
        FDE_result = np.mean(FDE_result)

    return FDE_result

def FDE(t_true, t_pred, batch_mean=True):
    t_true = t_true.permute(2, 1, 0)[:2].clone()
    t_pred = t_pred.permute(2, 1, 0)[:2].clone()

    xn_true = t_true[0][-1]
    xn_pred = t_pred[0][-1]
    yn_true = t_true[1][-1]
    yn_pred = t_pred[1][-1]

    x_diff = xn_true-xn_pred
    y_diff = yn_true-yn_pred

    sqroot = torch.sqrt(x_diff * x_diff + y_diff * y_diff)
    FDE_result = sqroot.cpu().detach().numpy()

    if batch_mean:
        FDE_result = np.mean(FDE_result)

    return FDE_result