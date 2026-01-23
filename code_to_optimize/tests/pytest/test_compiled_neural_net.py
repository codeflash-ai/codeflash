import torch
from code_to_optimize.unoptimized_neural_net import UnoptimizedNeuralNet

torch.manual_seed(42)
model = UnoptimizedNeuralNet(input_size=64, hidden_size=8, num_classes=5)
model = torch.compile(model)
batch_size = 4
input_tensor = torch.randn(batch_size, 64)
for _ in range(3):
    model(input_tensor)

def test_compiled_neural_net():
    output = model.forward(input_tensor)
    expected_output = torch.tensor([
        [3.5359490942882266e-12, 0.00012240608339197934, 5.034083642385667e-06, 0.999861478805542, 1.1137438377772924e-05],
        [0.9999979734420776, 1.7223709392055753e-06, 2.2974481785246303e-25, 6.0273253055243e-10, 3.6319553942121274e-07],
        [1.7603208387478864e-12, 1.0, 9.150994628726039e-09, 1.467387301078149e-12, 2.5555071392346917e-09],
        [9.112921543419361e-05, 0.005489135626703501, 0.6995948553085327, 0.21087759733200073, 0.08394725620746613]
    ])
    assert torch.allclose(output, expected_output)
