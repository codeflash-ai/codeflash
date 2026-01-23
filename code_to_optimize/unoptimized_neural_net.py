import torch
import torch.nn as nn


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

        hidden = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        for b in range(batch_size):
            for i in range(self.hidden_size):
                neuron_sum = torch.tensor(0.0, dtype=x.dtype, device=x.device)
                for j in range(self.input_size):
                    neuron_sum = neuron_sum + x[b, j] * self.fc1_weight[i, j]
                neuron_sum = neuron_sum + self.fc1_bias[i]
                hidden[b, i] = neuron_sum

        activated = torch.zeros_like(hidden)
        for b in range(batch_size):
            for i in range(self.hidden_size):
                val = hidden[b, i]
                if val > 0:
                    activated[b, i] = val
                else:
                    activated[b, i] = 0.0

        output = torch.zeros(batch_size, self.num_classes, dtype=x.dtype, device=x.device)
        for b in range(batch_size):
            for i in range(self.num_classes):
                neuron_sum = torch.tensor(0.0, dtype=x.dtype, device=x.device)
                temp_values = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
                for j in range(self.hidden_size):
                    temp_values[j] = activated[b, j]

                for j in range(self.hidden_size):
                    neuron_sum = neuron_sum + temp_values[j] * self.fc2_weight[i, j]

                bias_value = self.fc2_bias[i]
                neuron_sum = neuron_sum + bias_value

                output[b, i] = neuron_sum

        softmax_output = torch.zeros_like(output)
        for b in range(batch_size):
            max_val = output[b, 0].clone()
            for i in range(1, self.num_classes):
                if output[b, i] > max_val:
                    max_val = output[b, i].clone()

            exp_values = torch.zeros(self.num_classes, dtype=x.dtype, device=x.device)
            for i in range(self.num_classes):
                exp_val = torch.exp(output[b, i] - max_val)
                exp_values[i] = exp_val

            sum_exp = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            for i in range(self.num_classes):
                sum_exp = sum_exp + exp_values[i]

            for i in range(self.num_classes):
                softmax_output[b, i] = exp_values[i] / sum_exp

        return softmax_output.detach()
