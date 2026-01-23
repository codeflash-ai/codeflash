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

        hidden = []
        for b in range(batch_size):
            sample_output = []
            for i in range(self.hidden_size):
                neuron_sum = 0.0
                for j in range(self.input_size):
                    neuron_sum += x[b, j].item() * self.fc1_weight[i, j].item()
                neuron_sum += self.fc1_bias[i].item()
                sample_output.append(neuron_sum)
            hidden.append(sample_output)

        hidden = torch.tensor(hidden, dtype=x.dtype, device=x.device)

        activated = torch.zeros_like(hidden)
        for b in range(batch_size):
            for i in range(self.hidden_size):
                val = hidden[b, i].item()
                if val > 0:
                    activated[b, i] = val
                else:
                    activated[b, i] = 0.0

        output = []
        for b in range(batch_size):
            sample_output = []
            for i in range(self.num_classes):
                neuron_sum = 0.0
                temp_values = []
                for j in range(self.hidden_size):
                    temp_values.append(activated[b, j].item())

                for j in range(len(temp_values)):
                    neuron_sum += temp_values[j] * self.fc2_weight[i, j].item()

                bias_value = self.fc2_bias[i].item()
                neuron_sum += bias_value

                sample_output.append(neuron_sum)
            output.append(sample_output)

        output = torch.tensor(output, dtype=x.dtype, device=x.device)

        softmax_output = torch.zeros_like(output)
        for b in range(batch_size):
            max_val = output[b, 0].item()
            for i in range(1, self.num_classes):
                if output[b, i].item() > max_val:
                    max_val = output[b, i].item()

            exp_values = []
            for i in range(self.num_classes):
                exp_val = torch.exp(output[b, i] - max_val).item()
                exp_values.append(exp_val)

            sum_exp = sum(exp_values)

            for i in range(self.num_classes):
                softmax_output[b, i] = exp_values[i] / sum_exp

        return softmax_output
