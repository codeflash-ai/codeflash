from code_to_optimize.encode_python_string_to_c import _encodePythonStringToC


def test_empty_string():
    assert _encodePythonStringToC(b"") == '""'


def test_printable_characters():
    assert _encodePythonStringToC(b"hello world") == '"hello world"'


def test_special_characters():
    assert _encodePythonStringToC(b'hello\\world"') == '"hello\\134world\\042"'


def test_control_characters():
    assert _encodePythonStringToC(b"\t\n\r") == r'"\011\012\015"'


def test_non_printable_characters():
    assert _encodePythonStringToC(bytes([0, 1, 255])) == r'"\0\1\377"'


def test_mixed_content():
    assert _encodePythonStringToC(b"Line 1\nLine 2") == r'"Line 1\012Line 2"'


def test_adjacent_octal_characters():
    assert _encodePythonStringToC(b"\n123") == r'"\012" "123"'
