import pytest
from io import StringIO
from unittest.mock import patch
from cli.code_to_optimize.sample5 import main

@pytest.fixture
def mock_stdin(monkeypatch):
    def _mock_stdin(input_string):
        monkeypatch.setattr('sys.stdin', StringIO(input_string))
    return _mock_stdin

def test_main(mock_stdin, capsys):
    input_data = '2\nab\nba\n'
    expected_output = '4\n'
    mock_stdin(input_data)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output

def test_main_empty_string(mock_stdin, capsys):
    input_data = '1\n\n'
    expected_output = '0\n'
    mock_stdin(input_data)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output

def test_main_single_character(mock_stdin, capsys):
    input_data = '1\na\n'
    expected_output = '0\n'
    mock_stdin(input_data)
    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output
