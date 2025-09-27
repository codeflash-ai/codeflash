from pathlib import Path

from codeflash.code_utils.code_extractor import add_needed_imports_from_module, find_preexisting_objects
from codeflash.code_utils.code_replacer import replace_functions_and_add_imports

import tempfile
from codeflash.code_utils.code_extractor import resolve_star_import, DottedImportCollector
import libcst as cst
from codeflash.models.models import FunctionParent

def test_add_needed_imports_from_module0() -> None:
    src_module = '''import ast
import logging
import os
from typing import Union
import jedi
import tiktoken
from jedi.api.classes import Name
from pydantic.dataclasses import dataclass
from codeflash.code_utils.code_extractor import get_code, get_code_no_skeleton
from codeflash.code_utils.code_utils import path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize

def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if the given name belongs to the specified class."""
    if name.full_name and name.full_name.startswith(f"{name.module_name}.{class_name}."):
        return True
    return False

def heyjude() -> None:
    print("Hey Jude, don't make it bad")

def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname

@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Source:
    full_name: str
    definition: Name
    source_code: str
'''

    dst_module = """def heyjude() -> None:
    print("Hey Jude, don't make it bad")
"""

    expected = """def heyjude() -> None:
    print("Hey Jude, don't make it bad")
"""
    src_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    dst_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    project_root = Path("/home/roger/repos/codeflash")
    new_module = add_needed_imports_from_module(src_module, dst_module, src_path, dst_path, project_root)
    assert new_module == expected


def test_add_needed_imports_from_module() -> None:
    src_module = '''import ast
import logging
import os
from typing import Union

import jedi
import tiktoken
from jedi.api.classes import Name
from pydantic.dataclasses import dataclass

from codeflash.code_utils.code_extractor import get_code, get_code_no_skeleton
from codeflash.code_utils.code_utils import path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if the given name belongs to the specified class."""
    if name.full_name and name.full_name.startswith(f"{name.module_name}.{class_name}."):
        return True
    return False


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Source:
    full_name: str
    definition: Name
    source_code: str
'''

    dst_module = '''def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname
'''

    expected = '''from jedi.api.classes import Name

def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname
'''
    src_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    dst_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    project_root = Path("/home/roger/repos/codeflash")
    new_module = add_needed_imports_from_module(src_module, dst_module, src_path, dst_path, project_root)
    assert new_module == expected

def test_duplicated_imports() -> None:
    optim_code = '''from dataclasses import dataclass
from recce.adapter.base import BaseAdapter
from typing import Dict, List, Optional

@dataclass
class DbtAdapter(BaseAdapter):

    def build_parent_map(self, nodes: Dict, base: Optional[bool] = False) -> Dict[str, List[str]]:
        manifest = self.curr_manifest if base is False else self.base_manifest
        
        try:
            parent_map_source = manifest.parent_map
        except AttributeError:
            parent_map_source = manifest.to_dict()["parent_map"]

        node_ids = set(nodes)
        parent_map = {}
        for k, parents in parent_map_source.items():
            if k not in node_ids:
                continue
            parent_map[k] = [parent for parent in parents if parent in node_ids]

        return parent_map
'''

    original_code = '''import json
import logging
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from errno import ENOENT
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from recce.event import log_performance
from recce.exceptions import RecceException
from recce.util.cll import CLLPerformanceTracking, cll
from recce.util.lineage import (
    build_column_key,
    filter_dependency_maps,
    find_downstream,
    find_upstream,
)
from recce.util.perf_tracking import LineagePerfTracker

from ...tasks.profile import ProfileTask
from ...util.breaking import BreakingPerformanceTracking, parse_change_category

try:
    import agate
    import dbt.adapters.factory
    from dbt.contracts.state import PreviousState
except ImportError as e:
    print("Error: dbt module not found. Please install it by running:")
    print("pip install dbt-core dbt-<adapter>")
    raise e
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from recce.adapter.base import BaseAdapter
from recce.state import ArtifactsRoot

from ...models import RunType
from ...models.types import (
    CllColumn,
    CllData,
    CllNode,
    LineageDiff,
    NodeChange,
    NodeDiff,
)
from ...tasks import (
    HistogramDiffTask,
    ProfileDiffTask,
    QueryBaseTask,
    QueryDiffTask,
    QueryTask,
    RowCountDiffTask,
    RowCountTask,
    Task,
    TopKDiffTask,
    ValueDiffDetailTask,
    ValueDiffTask,
)
from .dbt_version import DbtVersion

@dataclass
class DbtAdapter(BaseAdapter):

    def build_parent_map(self, nodes: Dict, base: Optional[bool] = False) -> Dict[str, List[str]]:
        manifest = self.curr_manifest if base is False else self.base_manifest
        manifest_dict = manifest.to_dict()

        node_ids = nodes.keys()
        parent_map = {}
        for k, parents in manifest_dict["parent_map"].items():
            if k not in node_ids:
                continue
            parent_map[k] = [parent for parent in parents if parent in node_ids]

        return parent_map
'''
    expected = '''import json
import logging
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from errno import ENOENT
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from recce.event import log_performance
from recce.exceptions import RecceException
from recce.util.cll import CLLPerformanceTracking, cll
from recce.util.lineage import (
    build_column_key,
    filter_dependency_maps,
    find_downstream,
    find_upstream,
)
from recce.util.perf_tracking import LineagePerfTracker

from ...tasks.profile import ProfileTask
from ...util.breaking import BreakingPerformanceTracking, parse_change_category

try:
    import agate
    import dbt.adapters.factory
    from dbt.contracts.state import PreviousState
except ImportError as e:
    print("Error: dbt module not found. Please install it by running:")
    print("pip install dbt-core dbt-<adapter>")
    raise e
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from recce.adapter.base import BaseAdapter
from recce.state import ArtifactsRoot

from ...models import RunType
from ...models.types import (
    CllColumn,
    CllData,
    CllNode,
    LineageDiff,
    NodeChange,
    NodeDiff,
)
from ...tasks import (
    HistogramDiffTask,
    ProfileDiffTask,
    QueryBaseTask,
    QueryDiffTask,
    QueryTask,
    RowCountDiffTask,
    RowCountTask,
    Task,
    TopKDiffTask,
    ValueDiffDetailTask,
    ValueDiffTask,
)
from .dbt_version import DbtVersion

@dataclass
class DbtAdapter(BaseAdapter):

    def build_parent_map(self, nodes: Dict, base: Optional[bool] = False) -> Dict[str, List[str]]:
        manifest = self.curr_manifest if base is False else self.base_manifest
        
        try:
            parent_map_source = manifest.parent_map
        except AttributeError:
            parent_map_source = manifest.to_dict()["parent_map"]

        node_ids = set(nodes)
        parent_map = {}
        for k, parents in parent_map_source.items():
            if k not in node_ids:
                continue
            parent_map[k] = [parent for parent in parents if parent in node_ids]

        return parent_map
'''

    function_name: str = "DbtAdapter.build_parent_map"
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = find_preexisting_objects(original_code)
    new_code: str = replace_functions_and_add_imports(
        source_code=original_code,
        function_names=[function_name],
        optimized_code=optim_code,
        module_abspath=Path(__file__).resolve(),
        preexisting_objects=preexisting_objects,
        project_root_path=Path(__file__).resolve().parent.resolve(),
    )
    assert new_code == expected




def test_resolve_star_import_with_all_defined():
    """Test resolve_star_import when __all__ is explicitly defined."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        test_module = project_root / 'test_module.py'
        
        # Create a test module with __all__ definition
        test_module.write_text('''
__all__ = ['public_function', 'PublicClass']

def public_function():
    pass

def _private_function():
    pass

class PublicClass:
    pass

class AnotherPublicClass:
    """Not in __all__ so should be excluded."""
    pass
''')
        
        symbols = resolve_star_import('test_module', project_root)
        expected_symbols = {'public_function', 'PublicClass'}
        assert symbols == expected_symbols


def test_resolve_star_import_without_all_defined():
    """Test resolve_star_import when __all__ is not defined - should include all public symbols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        test_module = project_root / 'test_module.py'
        
        # Create a test module without __all__ definition
        test_module.write_text('''
def public_func():
    pass

def _private_func():
    pass

class PublicClass:
    pass

PUBLIC_VAR = 42
_private_var = 'secret'
''')
        
        symbols = resolve_star_import('test_module', project_root)
        expected_symbols = {'public_func', 'PublicClass', 'PUBLIC_VAR'}
        assert symbols == expected_symbols


def test_resolve_star_import_nonexistent_module():
    """Test resolve_star_import with non-existent module - should return empty set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        symbols = resolve_star_import('nonexistent_module', project_root)
        assert symbols == set()


def test_dotted_import_collector_skips_star_imports():
    """Test that DottedImportCollector correctly skips star imports."""
    code_with_star_import = '''
from typing import *
from pathlib import Path
from collections import defaultdict
import os
'''
    
    module = cst.parse_module(code_with_star_import)
    collector = DottedImportCollector()
    module.visit(collector)
    
    # Should collect regular imports but skip the star import
    expected_imports = {'collections.defaultdict', 'os', 'pathlib.Path'}
    assert collector.imports == expected_imports


def test_add_needed_imports_with_star_import_resolution():
    """Test add_needed_imports_from_module correctly handles star imports by resolving them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Create a source module that exports symbols
        src_module = project_root / 'source_module.py'
        src_module.write_text('''
__all__ = ['UtilFunction', 'HelperClass']

def UtilFunction():
    pass

class HelperClass:
    pass
''')
        
        # Create source code that uses star import
        src_code = '''
from source_module import *

def my_function():
    helper = HelperClass()
    UtilFunction()
    return helper
'''
        
        # Destination code that needs the imports resolved
        dst_code = '''
def my_function():
    helper = HelperClass()
    UtilFunction()
    return helper
'''
        
        src_path = project_root / 'src.py'
        dst_path = project_root / 'dst.py'
        src_path.write_text(src_code)
        
        result = add_needed_imports_from_module(
            src_code, dst_code, src_path, dst_path, project_root
        )
        
        # The result should have individual imports instead of star import
        expected_result = '''from source_module import HelperClass, UtilFunction

def my_function():
    helper = HelperClass()
    UtilFunction()
    return helper
'''
        assert result == expected_result
