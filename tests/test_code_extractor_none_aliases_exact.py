import tempfile
from pathlib import Path

from codeflash.languages.python.static_analysis.code_extractor import add_needed_imports_from_module


def test_add_needed_imports_with_none_aliases():
    source_code = """
import json
from typing import Dict as MyDict, Optional
from collections import defaultdict
    """

    target_code = """
def target_function():
    pass
    """

    expected_output = """
def target_function():
    pass
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_add_needed_imports_complex_aliases():
    source_code = """
import os
import sys as system
from typing import Dict, List as MyList, Optional as Opt
from collections import defaultdict as dd, Counter
from pathlib import Path
    """

    target_code = """
def my_function():
    return "test"
    """

    expected_output = """
def my_function():
    return "test"
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_add_needed_imports_with_usage():
    source_code = """
import json
from typing import Dict as MyDict, Optional
from collections import defaultdict

    """

    target_code = """
def target_function():
    data = json.loads('{"key": "value"}')
    my_dict: MyDict[str, str] = {}
    opt_value: Optional[str] = None
    dd = defaultdict(list)
    return data, my_dict, opt_value, dd
    """

    expected_output = """import json
from typing import Dict as MyDict, Optional
from collections import defaultdict

def target_function():
    data = json.loads('{"key": "value"}')
    my_dict: MyDict[str, str] = {}
    opt_value: Optional[str] = None
    dd = defaultdict(list)
    return data, my_dict, opt_value, dd
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        # Assert exact expected output
        assert result.strip() == expected_output.strip()


def test_litellm_router_style_imports():
    source_code = """
import asyncio
import copy
import json
from collections import defaultdict
from typing import Dict, List, Optional, Union
from litellm.types.utils import ModelInfo
from litellm.types.utils import ModelInfo as ModelMapInfo
    """

    target_code = '''
def target_function():
    """Target function for testing."""
    pass
    '''

    expected_output = '''
def target_function():
    """Target function for testing."""
    pass
    '''

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "complex_source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_edge_case_none_values_in_alias_pairs():
    source_code = """
from typing import Dict as MyDict, List, Optional as Opt
from collections import defaultdict, Counter as cnt
from pathlib import Path
    """

    target_code = """
def my_test_function():
    return "test"
    """

    expected_output = """
def my_test_function():
    return "test"
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "edge_case_source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_partial_import_usage():
    source_code = """
import os
import sys
from typing import Dict, List, Optional
from collections import defaultdict, Counter
    """

    target_code = """
def use_some_imports():
    path = os.path.join("a", "b") 
    my_dict: Dict[str, int] = {}
    counter = Counter([1, 2, 3])
    return path, my_dict, counter
    """

    expected_output = """import os
from collections import Counter
from typing import Dict

def use_some_imports():
    path = os.path.join("a", "b") 
    my_dict: Dict[str, int] = {}
    counter = Counter([1, 2, 3])
    return path, my_dict, counter
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_alias_handling():
    source_code = """
from typing import Dict as MyDict, List as MyList, Optional
from collections import defaultdict as dd, Counter
    """

    target_code = """
def test_aliases():
    d: MyDict[str, int] = {}
    lst: MyList[str] = []
    dd_instance = dd(list)
    return d, lst, dd_instance
    """

    expected_output = """from collections import defaultdict as dd
from typing import Dict as MyDict, List as MyList

def test_aliases():
    d: MyDict[str, int] = {}
    lst: MyList[str] = []
    dd_instance = dd(list)
    return d, lst, dd_instance
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        assert result.strip() == expected_output.strip()


def test_add_needed_imports_with_nonealiases():
    source_code = """
import json
from typing import Dict as MyDict, Optional
from collections import defaultdict

    """

    target_code = """
def target_function():
    pass
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "source.py"
        dst_path = temp_path / "target.py"

        src_path.write_text(source_code)
        dst_path.write_text(target_code)

        # This should not raise a TypeError
        result = add_needed_imports_from_module(
            src_module_code=source_code,
            dst_module_code=target_code,
            src_path=src_path,
            dst_path=dst_path,
            project_root=temp_path,
        )

        expected_output = """
def target_function():
    pass
    """
        assert result.strip() == expected_output.strip()
