import torch
def complex_activation(x):
    """A custom activation with many small operations - compile makes a huge difference"""
    # Many sequential element-wise ops create kernel launch overhead
    x = torch.sin(x)
    x = x * torch.cos(x)
    x = x + torch.exp(-x.abs())
    x = x / (1 + x.pow(2))
    x = torch.tanh(x) * torch.sigmoid(x)
    x = x - 0.5 * x.pow(3)
    return x