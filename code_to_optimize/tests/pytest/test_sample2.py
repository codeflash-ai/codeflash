import pytest
from io import StringIO
from unittest.mock import patch
from cli.code_to_optimize.sample2 import main

@pytest.fixture
def mock_inputs(monkeypatch):
    def _mock_inputs(input_strings):
        inputs = "\n".join(input_strings)
        monkeypatch.setattr('sys.stdin', StringIO(inputs))

    return _mock_inputs

def test_main(mock_inputs, capsys):
    test_input = [
        "2 2",  # h, w
        "..",    # row 1 of the chest
        ".."     # row 2 of the chest
    ]
    expected_output = "2\n"
    mock_inputs(test_input)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output

def test_main_large_input(mock_inputs, capsys):
    test_input = [
        "3 3",   # h, w
        "...",   # row 1 of the chest
        "...",   # row 2 of the chest
        "..."    # row 3 of the chest
    ]
    expected_output = "6\n"
    mock_inputs(test_input)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output
