import pytest
from io import StringIO
from unittest.mock import patch
from cli.code_to_optimize.sample3 import main

@pytest.fixture
def mock_inputs(monkeypatch):
    def _mock_inputs(input_strings):
        inputs = "\n".join(input_strings)
        monkeypatch.setattr('sys.stdin', StringIO(inputs))

    return _mock_inputs

def test_main_single(mock_inputs, capsys):
    test_input = [
        "1",    # n
        "1"     # a[0]
    ]
    expected_output = "1\n"
    mock_inputs(test_input)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output

def test_main_multiple(mock_inputs, capsys):
    test_input = [
        "5",    # n
        "1",    # a[0]
        "2",    # a[1]
        "2",    # a[2]
        "3",    # a[3]
        "3"     # a[4]
    ]
    expected_output = "3\n"
    mock_inputs(test_input)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output
