import torch
import torch.nn.functional as F
from torch import nn


class UnoptimizedNeuralNet(nn.Module):
    """A simple neural network with a highly unoptimized forward pass."""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(UnoptimizedNeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1_bias = nn.Parameter(torch.randn(hidden_size))
        self.fc2_weight = nn.Parameter(torch.randn(num_classes, hidden_size))
        self.fc2_bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        hidden = F.linear(x, self.fc1_weight, self.fc1_bias)
        activated = hidden.clamp_min(0.0)
        output = F.linear(activated, self.fc2_weight, self.fc2_bias)
        softmax_output = F.softmax(output, dim=1)

        return softmax_output.detach()
