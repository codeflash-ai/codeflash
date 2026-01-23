import torch
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

        x_det = x.detach()
        fc1_w = self.fc1_weight.detach().to(x.device, dtype=x.dtype)
        fc1_b = self.fc1_bias.detach().to(x.device, dtype=x.dtype)
        fc2_w = self.fc2_weight.detach().to(x.device, dtype=x.dtype)
        fc2_b = self.fc2_bias.detach().to(x.device, dtype=x.dtype)

        hidden = x_det.matmul(fc1_w.t()) + fc1_b

        activated = torch.clamp_min(hidden, 0.0)

        output = activated.matmul(fc2_w.t()) + fc2_b

        max_val, _ = output.max(dim=1, keepdim=True)
        exp_values = torch.exp(output - max_val)
        softmax_output = exp_values / exp_values.sum(dim=1, keepdim=True)

        return softmax_output
