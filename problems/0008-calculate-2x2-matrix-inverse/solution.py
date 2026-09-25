import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)

    if m.shape != (2, 2):
        raise ValueError("输入必须是 2×2 矩阵")

    try:
        return torch.linalg.inv(m)
    except RuntimeError:
        return None