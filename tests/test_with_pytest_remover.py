import ast

from codeflash.code_utils.with_pytest_remover import remove_pytest_raises

def test_remove_single_pytest_raises():
    original = """
def test_something():
    with pytest.raises(ValueError):
        raise ValueError('test')
"""
    expected = """
def test_something():
    raise ValueError('test')
"""
    tree = ast.parse(original)
    result = remove_pytest_raises(tree)
    # Convert result AST back to code and normalize whitespace
    result_code = ast.unparse(result).strip()
    assert result_code == expected.strip()


def test_remove_multiple_pytest_raises():
    original = """
def test_multiple():
    with pytest.raises(TypeError):
        int('abc')
    with pytest.raises(ValueError):
        int('')
"""
    expected = """
def test_multiple():
    int('abc')
    int('')
"""
    tree = ast.parse(original)
    result = remove_pytest_raises(tree)
    result_code = ast.unparse(result).strip()
    assert result_code == expected.strip()


def test_preserve_other_with_blocks():
    original = """
def test_mixed():
    with open('file.txt') as f:
        content = f.read()
    with pytest.raises(ValueError):
        int('abc')
    with contextlib.contextmanager():
        pass
"""
    expected = """
def test_mixed():
    with open('file.txt') as f:
        content = f.read()
    int('abc')
    with contextlib.contextmanager():
        pass
"""
    tree = ast.parse(original)
    result = remove_pytest_raises(tree)
    result_code = ast.unparse(result).strip()
    assert result_code == expected.strip()


def test_nested_with_blocks():
    original = """
def test_nested():
    with open('file.txt') as f:
        with pytest.raises(ValueError):
            int('abc')
"""
    expected = """
def test_nested():
    with open('file.txt') as f:
        int('abc')
"""
    tree = ast.parse(original)
    result = remove_pytest_raises(tree)
    result_code = ast.unparse(result).strip()
    assert result_code == expected.strip()


def test_no_pytest_raises():
    original = """
def test_normal():
    x = 1
    y = 2
    assert x + y == 3
"""
    tree = ast.parse(original)
    result = remove_pytest_raises(tree)
    result_code = ast.unparse(result).strip()
    assert result_code == original.strip()
