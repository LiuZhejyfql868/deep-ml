import torch
import torch.nn.functional as F

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    # Your code here
    z0 = torch.tensor(z)
    z1 = F.relu(z0)
    return z1
    
    pass
