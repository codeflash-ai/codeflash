import pytest
from io import StringIO
from unittest.mock import patch
from cli.code_to_optimize.sample1 import main


@pytest.fixture
def mock_inputs(monkeypatch):
    def _mock_inputs(input_strings):
        inputs = "\n".join(input_strings)
        monkeypatch.setattr("sys.stdin", StringIO(inputs))

    return _mock_inputs


def test_main(mock_inputs, capsys):
    test_input = ["5 3", "3", "1", "2"]  # n, m  # x1  # x2  # x3
    expected_output = "1\n2\n1\n1\n1\n"
    mock_inputs(test_input)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output
