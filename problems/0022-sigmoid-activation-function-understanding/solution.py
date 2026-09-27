import torch
import torch.nn as nn

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    x = torch.as_tensor(z,dtype=torch.float32)
    y=nn.Sigmoid()
    y1 = y(x)
    return round(y1.item(),4)
    