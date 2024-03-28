from codeflash.code_utils.formatter import format_code, sort_imports
import os
import tempfile


def test_remove_duplicate_imports():
    """
    Test that duplicate imports are removed when should_sort_imports is True.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"import os\nimport os\n")
        tmp_path = tmp.name

    new_code = sort_imports(imports_sort_cmd="isort", should_sort_imports=True, path=tmp_path)
    os.remove(tmp_path)
    assert new_code == "import os\n"


def test_remove_multiple_duplicate_imports():
    """
    Test that multiple duplicate imports are removed when should_sort_imports is True.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"import sys\nimport os\nimport sys\n")
        tmp_path = tmp.name

    new_code = sort_imports(imports_sort_cmd="isort", should_sort_imports=True, path=tmp_path)
    os.remove(tmp_path)
    assert new_code == "import os\nimport sys\n"


def test_sorting_imports():
    """
    Test that imports are sorted when should_sort_imports is True.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp_path = tmp.name

    new_code = sort_imports(imports_sort_cmd="isort", should_sort_imports=True, path=tmp_path)
    os.remove(tmp_path)
    assert new_code == "import os\nimport sys\nimport unittest\n"


def test_no_sorting_imports():
    """
    Test that imports are not sorted when should_sort_imports is False.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp_path = tmp.name

    new_code = sort_imports(imports_sort_cmd="isort", should_sort_imports=False, path=tmp_path)
    os.remove(tmp_path)
    assert new_code == "import sys\nimport unittest\nimport os\n"


def test_sort_imports_without_formatting():
    """
    Test that imports are sorted when formatting is disabled and should_sort_imports is True.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp_path = tmp.name

    new_code = format_code(
        formatter_cmd="disabled", imports_sort_cmd="isort", should_sort_imports=True, path=tmp_path
    )
    os.remove(tmp_path)
    assert new_code == "import os\nimport sys\nimport unittest\n"
