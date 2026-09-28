import torch
import torch.nn as nn
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    #features.tensor = torch.as_tensor(features,dtype=torch.float32)
    #labels.tensor = torch.as_tensor(labels, dtype=torch.float32)
    #weights.tensor = torch.as_tensor(weights,dtype = torch.float32)
    #bias.tensor = torch.as_tensor(bias, dtype=torch.float32)
    #activation = F.sigmoid()
    #logits = activation(weights.tensor * features.tensor + bias.tensor)
    #probs = round(logits, 4)
    #loss_function = F.mse_loss(probs, label.tensor,4)
    #forward() = ???
    #return (prob,MSE)

    # x = torch.tensor(features, dtype=torch.float32)
    # w = torch.tensor(weights, dtype=torch.float32)
    # y_true = torch.tensor(labels, dtype=torch.float32)

    # logits = torch.matmul(x, w) + bias
    # probabilities = torch.sigmoid(logits)

    # mse = F.mse_loss(probabilities, y_true)

    # rounded_probabilities = [
    #     round(p, 4) for p in probabilities.tolist()
    # ]
    # rounded_mse = round(mse.item(), 4)

    # return rounded_probabilities, rounded_mse


    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    weights_tensor = torch.as_tensor(weights, dtype=torch.float32)

    logits = torch.matmul(features_tensor, weights_tensor) + bias
    probs_tensor = torch.sigmoid(logits)
    mse_tensor = F.mse_loss(probs_tensor, labels_tensor)

    probs = [round(p, 4) for p in probs_tensor.tolist()]
    mse = round(mse_tensor.item(), 4)

    return probs, mse


    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    # pass