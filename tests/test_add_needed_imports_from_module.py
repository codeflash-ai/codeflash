import re
from pathlib import Path

from codeflash.code_utils.code_extractor import add_needed_imports_from_module


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
    src_module = '''from dataclasses import dataclass
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

    dst_module = '''import json
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
    src_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    dst_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    project_root = Path("/home/roger/repos/codeflash")
    new_module = add_needed_imports_from_module(src_module, dst_module, src_path, dst_path, project_root)

    matches = re.findall(r"^\s*from\s+recce\.adapter\.base\s+import\s+BaseAdapter\s*$", new_module, re.MULTILINE)
    assert len(matches) == 1, f"Expected 1 match for BaseAdapter import, but found {len(matches)}"
