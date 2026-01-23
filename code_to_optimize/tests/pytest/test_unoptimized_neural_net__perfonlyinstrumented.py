import gc
import os
import time

import numpy as np
import pytest
import torch

from code_to_optimize.unoptimized_neural_net import UnoptimizedNeuralNet


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f"{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}"
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value
'\nUnit tests for unoptimized PyTorch neural network implementation.\n\nTests run on CUDA devices only.\n'
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available - tests require CUDA')

def get_available_devices():
    """Return list of available PyTorch devices for testing."""
    return ['cuda']
DEVICES = get_available_devices()

def get_dtype(device):
    """Get the appropriate dtype for a device."""
    return torch.float32
FIXED_INPUT_SIZE = 784
FIXED_HIDDEN_SIZE = 128
FIXED_NUM_CLASSES = 10
FIXED_BATCH_SIZE = 4

class TestUnoptimizedNeuralNet:
    """Tests for the UnoptimizedNeuralNet class.

    All tests use a single fixed shape: (4, 784)
    """

    @pytest.mark.parametrize('device', DEVICES)
    def test_output_shape(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test that output shape is correct.'
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_output_shape', 'UnoptimizedNeuralNet.forward', '5', codeflash_loop_index, x)
        assert output.shape == (FIXED_BATCH_SIZE, FIXED_NUM_CLASSES)

    @pytest.mark.parametrize('device', DEVICES)
    def test_softmax_normalization(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test that output probabilities sum to 1.'
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_softmax_normalization', 'UnoptimizedNeuralNet.forward', '5', codeflash_loop_index, x)
        sums = output.sum(dim=1)
        np.testing.assert_array_almost_equal(sums.cpu().numpy(), np.ones(FIXED_BATCH_SIZE), decimal=5)

    @pytest.mark.parametrize('device', DEVICES)
    def test_output_range(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test that output values are in valid probability range [0, 1].'
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_output_range', 'UnoptimizedNeuralNet.forward', '5', codeflash_loop_index, x)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)

    @pytest.mark.parametrize('device', DEVICES)
    def test_zeros_input(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test with zero input.'
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        x = torch.zeros(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_zeros_input', 'UnoptimizedNeuralNet.forward', '4', codeflash_loop_index, x)
        assert output.shape == (FIXED_BATCH_SIZE, FIXED_NUM_CLASSES)
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize('device', DEVICES)
    def test_deterministic_output(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test that same input produces same output.'
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        model.eval()
        torch.manual_seed(100)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output1 = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_deterministic_output', 'UnoptimizedNeuralNet.forward', '7', codeflash_loop_index, x)
        output2 = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_deterministic_output', 'UnoptimizedNeuralNet.forward', '8', codeflash_loop_index, x)
        np.testing.assert_array_almost_equal(output1.detach().cpu().numpy(), output2.detach().cpu().numpy(), decimal=6)

    @pytest.mark.parametrize('device', DEVICES)
    def test_output_requires_grad_false(self, device):
        codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
        'Test that output has requires_grad=False.'
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(input_size=FIXED_INPUT_SIZE, hidden_size=FIXED_HIDDEN_SIZE, num_classes=FIXED_NUM_CLASSES)
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = codeflash_wrap(model.forward, 'code_to_optimize.tests.pytest.test_unoptimized_neural_net', 'TestUnoptimizedNeuralNet', 'test_output_requires_grad_false', 'UnoptimizedNeuralNet.forward', '5', codeflash_loop_index, x)
        assert output.requires_grad is False
