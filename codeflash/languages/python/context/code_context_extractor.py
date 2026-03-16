from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict, deque
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import encoded_tokens_len, get_qualified_name, path_belongs_to_site_packages
from codeflash.code_utils.config_consts import (
    OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    READ_WRITABLE_LIMIT_ERROR,
    TESTGEN_CONTEXT_TOKEN_LIMIT,
    TESTGEN_LIMIT_ERROR,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize  # noqa: TC001
from codeflash.languages.python.context.unused_definition_remover import (
    collect_top_level_defs_with_usages,
    get_section_names,
    is_assignment_used,
    recurse_sections,
    remove_unused_definitions_by_function_names,
)
from codeflash.languages.python.static_analysis.code_extractor import (
    add_needed_imports_from_module,
    find_preexisting_objects,
)
from codeflash.models.models import (
    CodeContextType,
    CodeOptimizationContext,
    CodeString,
    CodeStringsMarkdown,
    FunctionSource,
)

if TYPE_CHECKING:
    from jedi.api.classes import Name

    from codeflash.languages.base import DependencyResolver
    from codeflash.languages.python.context.unused_definition_remover import UsageInfo


def build_testgen_context(
    helpers_of_fto_dict: dict[Path, set[FunctionSource]],
    helpers_of_helpers_dict: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    *,
    remove_docstrings: bool = False,
    include_enrichment: bool = True,
    function_to_optimize: FunctionToOptimize | None = None,
) -> CodeStringsMarkdown:
    testgen_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=remove_docstrings,
        code_context_type=CodeContextType.TESTGEN,
    )

    if include_enrichment:
        enrichment = enrich_testgen_context(testgen_context, project_root_path)
        if enrichment.code_strings:
            testgen_context = CodeStringsMarkdown(code_strings=testgen_context.code_strings + enrichment.code_strings)

        if function_to_optimize is not None:
            result = _parse_and_collect_imports(testgen_context)
            existing_classes = collect_existing_class_names(result[0]) if result else set()
            constructor_stubs = extract_parameter_type_constructors(
                function_to_optimize, project_root_path, existing_classes
            )
            if constructor_stubs.code_strings:
                testgen_context = CodeStringsMarkdown(
                    code_strings=testgen_context.code_strings + constructor_stubs.code_strings
                )

    return testgen_context


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
    call_graph: DependencyResolver | None = None,
) -> CodeOptimizationContext:
    # Get FunctionSource representation of helpers of FTO
    fto_input = {function_to_optimize.file_path: {function_to_optimize.qualified_name}}
    if call_graph is not None:
        helpers_of_fto_dict, helpers_of_fto_list = call_graph.get_callees(fto_input)
    else:
        helpers_of_fto_dict, helpers_of_fto_list = get_function_sources_from_jedi(fto_input, project_root_path)

    # Add function to optimize into helpers of FTO dict, as they'll be processed together
    fto_as_function_source = get_function_to_optimize_as_function_source(function_to_optimize, project_root_path)
    helpers_of_fto_dict[function_to_optimize.file_path].add(fto_as_function_source)

    # Format data to search for helpers of helpers using get_function_sources_from_jedi
    helpers_of_fto_qualified_names_dict = {
        file_path: {source.qualified_name for source in sources} for file_path, sources in helpers_of_fto_dict.items()
    }

    # __init__ functions are automatically considered as helpers of FTO, so we add them to the dict (regardless of whether they exist)
    # This helps us to search for helpers of __init__ functions of classes that contain helpers of FTO
    for qualified_names in helpers_of_fto_qualified_names_dict.values():
        qualified_names.update({f"{qn.rsplit('.', 1)[0]}.__init__" for qn in qualified_names if "." in qn})

    helpers_of_helpers_dict, helpers_of_helpers_list = get_function_sources_from_jedi(
        helpers_of_fto_qualified_names_dict, project_root_path
    )

    # Extract code context for optimization
    final_read_writable_code = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        {},
        project_root_path,
        remove_docstrings=False,
        code_context_type=CodeContextType.READ_WRITABLE,
    )

    # Ensure the target file is first in the code blocks so the LLM knows which file to optimize
    target_relative = function_to_optimize.file_path.resolve().relative_to(project_root_path.resolve())
    target_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path == target_relative]
    other_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path != target_relative]
    if target_blocks:
        final_read_writable_code.code_strings = target_blocks + other_blocks

    read_only_code_markdown = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=False,
        code_context_type=CodeContextType.READ_ONLY,
    )
    hashing_code_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=True,
        code_context_type=CodeContextType.HASHING,
    )

    # Handle token limits
    final_read_writable_tokens = encoded_tokens_len(final_read_writable_code.markdown)
    if final_read_writable_tokens > optim_token_limit:
        raise ValueError(READ_WRITABLE_LIMIT_ERROR)

    # Setup preexisting objects for code replacer
    preexisting_objects = set(
        chain(
            *(find_preexisting_objects(codestring.code) for codestring in final_read_writable_code.code_strings),
            *(find_preexisting_objects(codestring.code) for codestring in read_only_code_markdown.code_strings),
        )
    )
    read_only_context_code = read_only_code_markdown.markdown

    # Progressive fallback for read-only context token limits
    read_only_tokens = encoded_tokens_len(read_only_context_code)
    if final_read_writable_tokens + read_only_tokens > optim_token_limit:
        logger.debug("Code context has exceeded token limit, removing docstrings from read-only code")
        read_only_code_no_docstrings = extract_code_markdown_context_from_files(
            helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, remove_docstrings=True
        )
        read_only_context_code = read_only_code_no_docstrings.markdown
        if final_read_writable_tokens + encoded_tokens_len(read_only_context_code) > optim_token_limit:
            logger.debug("Code context has exceeded token limit, removing read-only code")
            read_only_context_code = ""

    # Progressive fallback for testgen context token limits
    testgen_context = build_testgen_context(
        helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, function_to_optimize=function_to_optimize
    )

    if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
        logger.debug("Testgen context exceeded token limit, removing docstrings")
        testgen_context = build_testgen_context(
            helpers_of_fto_dict,
            helpers_of_helpers_dict,
            project_root_path,
            remove_docstrings=True,
            function_to_optimize=function_to_optimize,
        )

        if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
            logger.debug("Testgen context still exceeded token limit, removing enrichment")
            testgen_context = build_testgen_context(
                helpers_of_fto_dict,
                helpers_of_helpers_dict,
                project_root_path,
                remove_docstrings=True,
                include_enrichment=False,
            )

            if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
                raise ValueError(TESTGEN_LIMIT_ERROR)
    code_hash_context = hashing_code_context.markdown
    code_hash = hashlib.sha256(code_hash_context.encode("utf-8")).hexdigest()

    all_helper_fqns = list({fs.fully_qualified_name for fs in helpers_of_fto_list + helpers_of_helpers_list})

    return CodeOptimizationContext(
        testgen_context=testgen_context,
        read_writable_code=final_read_writable_code,
        read_only_context_code=read_only_context_code,
        hashing_code_context=code_hash_context,
        hashing_code_context_hash=code_hash,
        helper_functions=helpers_of_fto_list,
        testgen_helper_fqns=all_helper_fqns,
        preexisting_objects=preexisting_objects,
    )


def process_file_context(
    file_path: Path,
    primary_qualified_names: set[str],
    secondary_qualified_names: set[str],
    code_context_type: CodeContextType,
    remove_docstrings: bool,
    project_root_path: Path,
    helper_functions: list[FunctionSource],
) -> CodeString | None:
    try:
        original_code = file_path.read_text("utf8")
    except Exception as e:
        logger.exception(f"Error while parsing {file_path}: {e}")
        return None

    try:
        all_names = primary_qualified_names | secondary_qualified_names
        code_without_unused_defs = remove_unused_definitions_by_function_names(original_code, all_names)
        pruned_module = parse_code_and_prune_cst(
            code_without_unused_defs,
            code_context_type,
            primary_qualified_names,
            secondary_qualified_names,
            remove_docstrings,
        )
    except ValueError as e:
        logger.debug(f"Error while getting read-only code: {e}")
        return None

    if pruned_module.code.strip():
        if code_context_type == CodeContextType.HASHING:
            code_context = ast.unparse(ast.parse(pruned_module.code))
        else:
            code_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=pruned_module,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions=helper_functions,
            )
        try:
            relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        except ValueError:
            relative_path = file_path
        return CodeString(code=code_context, file_path=relative_path)
    return None


def extract_code_markdown_context_from_files(
    helpers_of_fto: dict[Path, set[FunctionSource]],
    helpers_of_helpers: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    remove_docstrings: bool = False,
    code_context_type: CodeContextType = CodeContextType.READ_ONLY,
) -> CodeStringsMarkdown:
    """Extract code context from files containing target functions and their helpers, formatting them as markdown.

    This function processes two sets of files:
    1. Files containing the function to optimize (fto) and their first-degree helpers
    2. Files containing only helpers of helpers (with no overlap with the first set)

    For each file, it extracts relevant code based on the specified context type, adds necessary
    imports, and combines them into a structured markdown format.

    Args:
    ----
        helpers_of_fto: Dictionary mapping file paths to sets of Function Sources of function to optimize and its helpers
        helpers_of_helpers: Dictionary mapping file paths to sets of Function Sources of helpers of helper functions
        project_root_path: Root path of the project
        remove_docstrings: Whether to remove docstrings from the extracted code
        code_context_type: Type of code context to extract (READ_ONLY, READ_WRITABLE, or TESTGEN)

    Returns:
    -------
        CodeStringsMarkdown containing the extracted code context with necessary imports,
        formatted for inclusion in markdown

    """
    # Rearrange to remove overlaps, so we only access each file path once
    helpers_of_helpers_no_overlap = defaultdict(set)
    for file_path, function_sources in helpers_of_helpers.items():
        if file_path in helpers_of_fto:
            # Remove duplicates within the same file path, in case a helper of helper is also a helper of fto
            helpers_of_helpers[file_path] -= helpers_of_fto[file_path]
        else:
            helpers_of_helpers_no_overlap[file_path] = function_sources
    code_context_markdown = CodeStringsMarkdown()
    # Extract code from file paths that contain fto and first degree helpers. helpers of helpers may also be included if they are in the same files
    for file_path, function_sources in helpers_of_fto.items():
        qualified_function_names = {func.qualified_name for func in function_sources}
        helpers_of_helpers_qualified_names = {func.qualified_name for func in helpers_of_helpers.get(file_path, set())}
        helper_functions = list(helpers_of_fto.get(file_path, set()) | helpers_of_helpers.get(file_path, set()))

        result = process_file_context(
            file_path=file_path,
            primary_qualified_names=qualified_function_names,
            secondary_qualified_names=helpers_of_helpers_qualified_names,
            code_context_type=code_context_type,
            remove_docstrings=remove_docstrings,
            project_root_path=project_root_path,
            helper_functions=helper_functions,
        )

        if result is not None:
            code_context_markdown.code_strings.append(result)
    # Extract code from file paths containing helpers of helpers
    for file_path, helper_function_sources in helpers_of_helpers_no_overlap.items():
        qualified_helper_function_names = {func.qualified_name for func in helper_function_sources}
        helper_functions = list(helpers_of_helpers_no_overlap.get(file_path, set()))

        result = process_file_context(
            file_path=file_path,
            primary_qualified_names=set(),
            secondary_qualified_names=qualified_helper_function_names,
            code_context_type=code_context_type,
            remove_docstrings=remove_docstrings,
            project_root_path=project_root_path,
            helper_functions=helper_functions,
        )

        if result is not None:
            code_context_markdown.code_strings.append(result)
    return code_context_markdown


def get_function_to_optimize_as_function_source(
    function_to_optimize: FunctionToOptimize, project_root_path: Path
) -> FunctionSource:
    import jedi

    # Use jedi to find function to optimize
    script = jedi.Script(path=function_to_optimize.file_path, project=jedi.Project(path=project_root_path))

    # Get all names in the file
    names = script.get_names(all_scopes=True, definitions=True, references=False)

    # Find the name that matches our function
    for name in names:
        try:
            if (
                name.type == "function"
                and name.full_name
                and name.name == function_to_optimize.function_name
                and name.full_name.startswith(name.module_name)
                and get_qualified_name(name.module_name, name.full_name) == function_to_optimize.qualified_name
            ):
                return FunctionSource(
                    file_path=function_to_optimize.file_path,
                    qualified_name=function_to_optimize.qualified_name,
                    fully_qualified_name=name.full_name,
                    only_function_name=name.name,
                    source_code=name.get_line_code(),
                )
        except Exception as e:
            logger.exception(f"Error while getting function source: {e}")
            continue
    raise ValueError(
        f"Could not find function {function_to_optimize.function_name} in {function_to_optimize.file_path}"  # noqa: EM102
    )


def get_function_sources_from_jedi(
    file_path_to_qualified_function_names: dict[Path, set[str]], project_root_path: Path
) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:
    import jedi

    file_path_to_function_source = defaultdict(set)
    function_source_list: list[FunctionSource] = []
    for file_path, qualified_function_names in file_path_to_qualified_function_names.items():
        script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
        file_refs = script.get_names(all_scopes=True, definitions=False, references=True)

        for qualified_function_name in qualified_function_names:
            names = [
                ref
                for ref in file_refs
                if ref.full_name and belongs_to_function_qualified(ref, qualified_function_name)
            ]
            for name in names:
                try:
                    definitions: list[Name] = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception:
                    logger.debug(f"Error while getting definitions for {qualified_function_name}")
                    definitions = []
                if definitions:
                    # TODO: there can be multiple definitions, see how to handle such cases
                    definition = definitions[0]
                    definition_path = definition.module_path
                    if definition_path is not None:
                        try:
                            rel = definition_path.resolve().relative_to(project_root_path.resolve())
                            definition_path = project_root_path / rel
                        except ValueError:
                            pass

                    # The definition is part of this project and not defined within the original function
                    is_valid_definition = (
                        definition_path is not None
                        and not path_belongs_to_site_packages(definition_path)
                        and str(definition_path).startswith(str(project_root_path) + os.sep)
                        and definition.full_name
                        and not belongs_to_function_qualified(definition, qualified_function_name)
                        and definition.full_name.startswith(definition.module_name)
                    )
                    if is_valid_definition and definition.type in ("function", "class", "statement"):
                        if definition.type == "function":
                            fqn = definition.full_name
                            func_name = definition.name
                        elif definition.type == "class":
                            fqn = f"{definition.full_name}.__init__"
                            func_name = "__init__"
                        else:
                            fqn = definition.full_name
                            func_name = definition.name
                        qualified_name = get_qualified_name(definition.module_name, fqn)
                        # Avoid nested functions or classes. Only class.function is allowed
                        if len(qualified_name.split(".")) <= 2:
                            function_source = FunctionSource(
                                file_path=definition_path,
                                qualified_name=qualified_name,
                                fully_qualified_name=fqn,
                                only_function_name=func_name,
                                source_code=definition.get_line_code(),
                                definition_type=definition.type,
                            )
                            file_path_to_function_source[definition_path].add(function_source)
                            function_source_list.append(function_source)

    return file_path_to_function_source, function_source_list


def _parse_and_collect_imports(code_context: CodeStringsMarkdown) -> tuple[ast.Module, dict[str, str]] | None:
    all_code = "\n".join(cs.code for cs in code_context.code_strings)
    try:
        tree = ast.parse(all_code)
    except SyntaxError:
        return None
    imported_names: dict[str, str] = {}

    # Directly iterate over the module body and nested structures instead of ast.walk
    # This avoids traversing every single node in the tree
    def collect_imports(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        imported_name = alias.asname if alias.asname else alias.name
                        imported_names[imported_name] = node.module
            # Recursively check nested structures (function defs, class defs, if statements, etc.)
            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.ExceptHandler,
                ),
            ):
                if hasattr(node, "body"):
                    collect_imports(node.body)
                if hasattr(node, "orelse"):
                    collect_imports(node.orelse)
                if hasattr(node, "finalbody"):
                    collect_imports(node.finalbody)
                if hasattr(node, "handlers"):
                    for handler in node.handlers:
                        collect_imports(handler.body)
            # Handle match/case statements (Python 3.10+)
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                for case in node.cases:
                    collect_imports(case.body)

    collect_imports(tree.body)
    return tree, imported_names


def collect_existing_class_names(tree: ast.Module) -> set[str]:
    class_names = set()
    stack = list(tree.body)

    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            stack.extend(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.body)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            stack.extend(node.body)
            if hasattr(node, "orelse"):
                stack.extend(node.orelse)
        elif isinstance(node, ast.Try):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
            for handler in node.handlers:
                stack.extend(handler.body)

    return class_names


BUILTIN_AND_TYPING_NAMES = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "type",
        "object",
        "None",
        "NoneType",
        "Ellipsis",
        "NotImplemented",
        "memoryview",
        "range",
        "slice",
        "property",
        "classmethod",
        "staticmethod",
        "super",
        "Optional",
        "Union",
        "Any",
        "List",
        "Dict",
        "Set",
        "FrozenSet",
        "Tuple",
        "Type",
        "Callable",
        "Iterator",
        "Generator",
        "Coroutine",
        "AsyncGenerator",
        "AsyncIterator",
        "Iterable",
        "AsyncIterable",
        "Sequence",
        "MutableSequence",
        "Mapping",
        "MutableMapping",
        "Collection",
        "Awaitable",
        "Literal",
        "Final",
        "ClassVar",
        "TypeVar",
        "TypeAlias",
        "ParamSpec",
        "Concatenate",
        "Annotated",
        "TypeGuard",
        "Self",
        "Unpack",
        "TypeVarTuple",
        "Never",
        "NoReturn",
        "SupportsInt",
        "SupportsFloat",
        "SupportsComplex",
        "SupportsBytes",
        "SupportsAbs",
        "SupportsRound",
        "IO",
        "TextIO",
        "BinaryIO",
        "Pattern",
        "Match",
    }
)


def collect_type_names_from_annotation(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Subscript):
        names = collect_type_names_from_annotation(node.value)
        names |= collect_type_names_from_annotation(node.slice)
        return names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return collect_type_names_from_annotation(node.left) | collect_type_names_from_annotation(node.right)
    if isinstance(node, ast.Tuple):
        names = set[str]()
        for elt in node.elts:
            names |= collect_type_names_from_annotation(elt)
        return names
    return set()


MAX_RAW_PROJECT_CLASS_BODY_ITEMS = 8
MAX_RAW_PROJECT_CLASS_LINES = 40


def _get_expr_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent_name = _get_expr_name(node.value)
        return node.attr if parent_name is None else f"{parent_name}.{node.attr}"
    if isinstance(node, ast.Call):
        return _get_expr_name(node.func)
    return None


def _collect_import_aliases(module_tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname if alias.asname else alias.name.split(".")[0]
                aliases[bound_name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bound_name = alias.asname if alias.asname else alias.name
                aliases[bound_name] = f"{node.module}.{alias.name}"
    return aliases


def _find_class_node_by_name(class_name: str, module_tree: ast.Module) -> ast.ClassDef | None:
    # Use a deque-based BFS to find the first matching ClassDef (preserves ast.walk order)
    q: deque[ast.AST] = deque([module_tree])
    while q:
        candidate = q.popleft()
        if isinstance(candidate, ast.ClassDef) and candidate.name == class_name:
            return candidate
        q.extend(ast.iter_child_nodes(candidate))
    return None


def _expr_matches_name(node: ast.AST | None, import_aliases: dict[str, str], suffix: str) -> bool:
    expr_name = _get_expr_name(node)
    if expr_name is None:
        return False
    if expr_name == suffix or expr_name.endswith(f".{suffix}"):
        return True
    resolved_name = import_aliases.get(expr_name)
    return resolved_name is not None and (resolved_name == suffix or resolved_name.endswith(f".{suffix}"))


def _get_node_source(node: ast.AST | None, module_source: str, fallback: str = "...") -> str:
    if node is None:
        return fallback
    source_segment = ast.get_source_segment(module_source, node)
    if source_segment is not None:
        return source_segment
    try:
        return ast.unparse(node)
    except Exception:
        return fallback


def _bool_literal(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _is_namedtuple_class(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> bool:
    return any(_expr_matches_name(base, import_aliases, "NamedTuple") for base in class_node.bases)


def _get_dataclass_config(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> tuple[bool, bool, bool]:
    for decorator in class_node.decorator_list:
        if not _expr_matches_name(decorator, import_aliases, "dataclass"):
            continue
        init_enabled = True
        kw_only = False
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                literal_value = _bool_literal(keyword.value)
                if literal_value is None:
                    continue
                if keyword.arg == "init":
                    init_enabled = literal_value
                elif keyword.arg == "kw_only":
                    kw_only = literal_value
        return True, init_enabled, kw_only
    return False, False, False


def _is_classvar_annotation(annotation: ast.expr, import_aliases: dict[str, str]) -> bool:
    annotation_root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return _expr_matches_name(annotation_root, import_aliases, "ClassVar")


def _is_project_path(module_path: Path, project_root_path: Path) -> bool:
    return str(module_path.resolve()).startswith(str(project_root_path.resolve()) + os.sep)


def _get_class_start_line(class_node: ast.ClassDef) -> int:
    start_line = class_node.lineno
    if class_node.decorator_list:
        for decorator in class_node.decorator_list:
            start_line = min(start_line, decorator.lineno)
    return start_line


def _class_has_explicit_init(class_node: ast.ClassDef) -> bool:
    return any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__" for item in class_node.body)


def _collect_synthetic_constructor_type_names(
    class_node: ast.ClassDef, import_aliases: dict[str, str]
) -> set[str]:
    is_dataclass, dataclass_init_enabled, _ = _get_dataclass_config(class_node, import_aliases)
    if not _is_namedtuple_class(class_node, import_aliases) and not is_dataclass:
        return set()
    if is_dataclass and not dataclass_init_enabled:
        return set()

    names = set[str]()
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name) or item.annotation is None:
            continue
        if _is_classvar_annotation(item.annotation, import_aliases):
            continue

        include_in_init = True
        if isinstance(item.value, ast.Call) and _expr_matches_name(item.value.func, import_aliases, "field"):
            for keyword in item.value.keywords:
                if keyword.arg != "init":
                    continue
                literal_value = _bool_literal(keyword.value)
                if literal_value is not None:
                    include_in_init = literal_value
                break

        if include_in_init:
            names |= collect_type_names_from_annotation(item.annotation)

    return names


def _extract_synthetic_init_parameters(
    class_node: ast.ClassDef, module_source: str, import_aliases: dict[str, str], *, kw_only_by_default: bool
) -> list[tuple[str, str, str | None, bool]]:
    parameters: list[tuple[str, str, str | None, bool]] = []
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if _is_classvar_annotation(item.annotation, import_aliases):
            continue

        include_in_init = True
        kw_only = kw_only_by_default
        default_value: str | None = None
        if item.value is not None:
            if isinstance(item.value, ast.Call) and _expr_matches_name(item.value.func, import_aliases, "field"):
                for keyword in item.value.keywords:
                    if keyword.arg == "init":
                        literal_value = _bool_literal(keyword.value)
                        if literal_value is not None:
                            include_in_init = literal_value
                    elif keyword.arg == "kw_only":
                        literal_value = _bool_literal(keyword.value)
                        if literal_value is not None:
                            kw_only = literal_value
                    elif keyword.arg == "default":
                        default_value = _get_node_source(keyword.value, module_source)
                    elif keyword.arg == "default_factory":
                        # Default factories still imply an optional constructor parameter, but
                        # the generated __init__ does not use the field() call directly.
                        default_value = "..."
            else:
                default_value = _get_node_source(item.value, module_source)

        if not include_in_init:
            continue

        parameters.append((item.target.id, _get_node_source(item.annotation, module_source, "Any"), default_value, kw_only))
    return parameters


def _build_synthetic_init_stub(
    class_node: ast.ClassDef, module_source: str, import_aliases: dict[str, str]
) -> str | None:
    is_namedtuple = _is_namedtuple_class(class_node, import_aliases)
    is_dataclass, dataclass_init_enabled, dataclass_kw_only = _get_dataclass_config(class_node, import_aliases)
    if not is_namedtuple and not is_dataclass:
        return None
    if is_dataclass and not dataclass_init_enabled:
        return None

    parameters = _extract_synthetic_init_parameters(
        class_node,
        module_source,
        import_aliases,
        kw_only_by_default=dataclass_kw_only,
    )
    if not parameters:
        return None

    signature_parts = ["self"]
    inserted_kw_only_marker = False
    for param_name, annotation_source, default_value, kw_only in parameters:
        if kw_only and not inserted_kw_only_marker:
            signature_parts.append("*")
            inserted_kw_only_marker = True
        part = f"{param_name}: {annotation_source}"
        if default_value is not None:
            part += f" = {default_value}"
        signature_parts.append(part)

    signature = ", ".join(signature_parts)
    return f"    def __init__({signature}):\n        ..."


def _extract_function_stub_snippet(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef, module_lines: list[str]
) -> str:
    start_line = fn_node.lineno
    if fn_node.decorator_list:
        for decorator in fn_node.decorator_list:
            start_line = min(start_line, decorator.lineno)
    return "\n".join(module_lines[start_line - 1 : fn_node.end_lineno])


def _extract_raw_class_context(class_node: ast.ClassDef, module_source: str, module_tree: ast.Module) -> str:
    class_source = "\n".join(module_source.splitlines()[_get_class_start_line(class_node) - 1 : class_node.end_lineno])
    needed_imports = extract_imports_for_class(module_tree, class_node, module_source)
    if needed_imports:
        return f"{needed_imports}\n\n{class_source}"
    return class_source


def _has_non_property_method_decorator(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef, import_aliases: dict[str, str]
) -> bool:
    for decorator in fn_node.decorator_list:
        if _expr_matches_name(decorator, import_aliases, "property"):
            continue
        decorator_name = _get_expr_name(decorator)
        if decorator_name is not None and decorator_name.endswith(".setter"):
            continue
        if decorator_name is not None and decorator_name.endswith(".deleter"):
            continue
        return True
    return False


def _has_descriptor_like_class_fields(class_node: ast.ClassDef) -> bool:
    for item in class_node.body:
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
            return True
        if isinstance(item, ast.AnnAssign) and isinstance(item.value, ast.Call):
            return True
    return False


def _should_use_raw_project_class_context(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> bool:
    start_line = _get_class_start_line(class_node)
    class_line_count = class_node.end_lineno - start_line + 1
    is_small = class_line_count <= MAX_RAW_PROJECT_CLASS_LINES and len(class_node.body) <= MAX_RAW_PROJECT_CLASS_BODY_ITEMS

    if is_small and _class_has_explicit_init(class_node):
        return True
    if _is_namedtuple_class(class_node, import_aliases):
        return True
    is_dataclass, _, _ = _get_dataclass_config(class_node, import_aliases)
    if is_dataclass:
        return True
    if class_node.decorator_list:
        return True
    if _has_descriptor_like_class_fields(class_node):
        return True

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _has_non_property_method_decorator(
            item, import_aliases
        ):
            return True

    return False


def extract_init_stub_from_class(class_name: str, module_source: str, module_tree: ast.Module) -> str | None:
    class_node = _find_class_node_by_name(class_name, module_tree)

    if class_node is None:
        return None

    lines = module_source.splitlines()
    import_aliases = _collect_import_aliases(module_tree)
    explicit_init_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    support_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                explicit_init_nodes.append(item)
                support_nodes.append(item)
                continue
            if item.name == "__post_init__":
                support_nodes.append(item)
                continue
            # Check decorators explicitly to avoid generator overhead
            for d in item.decorator_list:
                if (isinstance(d, ast.Name) and d.id == "property") or (
                    isinstance(d, ast.Attribute) and d.attr == "property"
                ):
                    support_nodes.append(item)
                    break

    snippets: list[str] = []
    if explicit_init_nodes:
        for fn_node in support_nodes:
            snippets.append(_extract_function_stub_snippet(fn_node, lines))
    else:
        synthetic_init = _build_synthetic_init_stub(class_node, module_source, import_aliases)
        if synthetic_init is not None:
            snippets.append(synthetic_init)
        for fn_node in support_nodes:
            snippets.append(_extract_function_stub_snippet(fn_node, lines))

    if not snippets:
        return None

    return f"class {class_name}:\n" + "\n".join(snippets)


def _get_module_source_and_tree(
    module_path: Path, module_cache: dict[Path, tuple[str, ast.Module]]
) -> tuple[str, ast.Module] | None:
    if module_path in module_cache:
        return module_cache[module_path]
    try:
        module_source = module_path.read_text(encoding="utf-8")
        module_tree = ast.parse(module_source)
    except Exception:
        return None
    module_cache[module_path] = (module_source, module_tree)
    return module_source, module_tree


def _resolve_imported_class_reference(
    base_expr_name: str,
    current_module_tree: ast.Module,
    current_module_path: Path,
    project_root_path: Path,
    module_cache: dict[Path, tuple[str, ast.Module]],
) -> tuple[str, Path] | None:
    import jedi

    import_aliases = _collect_import_aliases(current_module_tree)
    class_name = base_expr_name.rsplit(".", 1)[-1]
    if "." not in base_expr_name and _find_class_node_by_name(class_name, current_module_tree) is not None:
        return class_name, current_module_path

    resolved_name = base_expr_name
    if base_expr_name in import_aliases:
        resolved_name = import_aliases[base_expr_name]
    elif "." in base_expr_name:
        head, tail = base_expr_name.split(".", 1)
        if head in import_aliases:
            resolved_name = f"{import_aliases[head]}.{tail}"

    if "." not in resolved_name:
        return None

    module_name, class_name = resolved_name.rsplit(".", 1)
    try:
        script_code = f"from {module_name} import {class_name}"
        script = jedi.Script(script_code, project=jedi.Project(path=project_root_path))
        definitions = script.goto(1, len(f"from {module_name} import ") + len(class_name), follow_imports=True)
    except Exception:
        return None

    if not definitions or definitions[0].module_path is None:
        return None
    module_path = definitions[0].module_path
    if not _is_project_path(module_path, project_root_path):
        return None
    if _get_module_source_and_tree(module_path, module_cache) is None:
        return None
    return class_name, module_path


def _append_project_class_context(
    class_name: str,
    module_path: Path,
    project_root_path: Path,
    module_cache: dict[Path, tuple[str, ast.Module]],
    existing_class_names: set[str],
    emitted_classes: set[tuple[Path, str]],
    emitted_class_names: set[str],
    code_strings: list[CodeString],
) -> bool:
    module_result = _get_module_source_and_tree(module_path, module_cache)
    if module_result is None:
        return False
    module_source, module_tree = module_result
    class_node = _find_class_node_by_name(class_name, module_tree)
    if class_node is None:
        return False

    class_key = (module_path, class_name)
    if class_key in emitted_classes or class_name in existing_class_names:
        return True

    for base in class_node.bases:
        base_expr_name = _get_expr_name(base)
        if base_expr_name is None:
            continue
        resolved = _resolve_imported_class_reference(
            base_expr_name,
            module_tree,
            module_path,
            project_root_path,
            module_cache,
        )
        if resolved is None:
            continue
        base_name, base_module_path = resolved
        if base_name in existing_class_names:
            continue
        _append_project_class_context(
            base_name,
            base_module_path,
            project_root_path,
            module_cache,
            existing_class_names,
            emitted_classes,
            emitted_class_names,
            code_strings,
        )

    code_strings.append(CodeString(code=_extract_raw_class_context(class_node, module_source, module_tree), file_path=module_path))
    emitted_classes.add(class_key)
    emitted_class_names.add(class_name)
    return True


def extract_parameter_type_constructors(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, existing_class_names: set[str]
) -> CodeStringsMarkdown:
    import jedi

    try:
        source = function_to_optimize.file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return CodeStringsMarkdown(code_strings=[])

    func_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_to_optimize.function_name
        ):
            if function_to_optimize.starting_line is not None and node.lineno != function_to_optimize.starting_line:
                continue
            func_node = node
            break
    if func_node is None:
        return CodeStringsMarkdown(code_strings=[])

    type_names: set[str] = set()
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        type_names |= collect_type_names_from_annotation(arg.annotation)
    if func_node.args.vararg:
        type_names |= collect_type_names_from_annotation(func_node.args.vararg.annotation)
    if func_node.args.kwarg:
        type_names |= collect_type_names_from_annotation(func_node.args.kwarg.annotation)

    # Scan function body for isinstance(x, SomeType) and type(x) is/== SomeType patterns
    for body_node in ast.walk(func_node):
        if (
            isinstance(body_node, ast.Call)
            and isinstance(body_node.func, ast.Name)
            and body_node.func.id == "isinstance"
        ):
            if len(body_node.args) >= 2:
                second_arg = body_node.args[1]
                if isinstance(second_arg, ast.Name):
                    type_names.add(second_arg.id)
                elif isinstance(second_arg, ast.Tuple):
                    for elt in second_arg.elts:
                        if isinstance(elt, ast.Name):
                            type_names.add(elt.id)
        elif isinstance(body_node, ast.Compare):
            # type(x) is/== SomeType
            if (
                isinstance(body_node.left, ast.Call)
                and isinstance(body_node.left.func, ast.Name)
                and body_node.left.func.id == "type"
            ):
                for comparator in body_node.comparators:
                    if isinstance(comparator, ast.Name):
                        type_names.add(comparator.id)

    # Collect base class names from enclosing class (if this is a method)
    if function_to_optimize.class_name is not None:
        for top_node in ast.walk(tree):
            if isinstance(top_node, ast.ClassDef) and top_node.name == function_to_optimize.class_name:
                for base in top_node.bases:
                    if isinstance(base, ast.Name):
                        type_names.add(base.id)
                break

    type_names -= BUILTIN_AND_TYPING_NAMES
    type_names -= existing_class_names
    if not type_names:
        return CodeStringsMarkdown(code_strings=[])

    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_map[name] = node.module

    code_strings: list[CodeString] = []
    module_cache: dict[Path, tuple[str, ast.Module]] = {}
    emitted_classes: set[tuple[Path, str]] = set()
    emitted_class_names: set[str] = set()

    def append_type_context(type_name: str, module_name: str, *, transitive: bool = False) -> None:
        try:
            script_code = f"from {module_name} import {type_name}"
            script = jedi.Script(script_code, project=jedi.Project(path=project_root_path))
            definitions = script.goto(1, len(f"from {module_name} import ") + len(type_name), follow_imports=True)
            if not definitions:
                return

            module_path = definitions[0].module_path
            if not module_path:
                return
            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = _is_project_path(module_path, project_root_path)
            is_third_party = "site-packages" in module_str
            if transitive and not is_project and not is_third_party:
                return

            module_result = _get_module_source_and_tree(module_path, module_cache)
            if module_result is None:
                return
            mod_source, mod_tree = module_result

            class_key = (module_path, type_name)
            if class_key in emitted_classes or type_name in existing_class_names:
                return

            class_node = _find_class_node_by_name(type_name, mod_tree)
            if class_node is not None and is_project:
                import_aliases = _collect_import_aliases(mod_tree)
                if _should_use_raw_project_class_context(class_node, import_aliases):
                    if _append_project_class_context(
                        type_name,
                        module_path,
                        project_root_path,
                        module_cache,
                        existing_class_names,
                        emitted_classes,
                        emitted_class_names,
                        code_strings,
                    ):
                        return

            stub = extract_init_stub_from_class(type_name, mod_source, mod_tree)
            if stub:
                code_strings.append(CodeString(code=stub, file_path=module_path))
                emitted_classes.add(class_key)
                emitted_class_names.add(type_name)
        except Exception:
            if transitive:
                logger.debug(f"Error extracting transitive constructor stub for {type_name} from {module_name}")
            else:
                logger.debug(f"Error extracting constructor stub for {type_name} from {module_name}")

    for type_name in sorted(type_names):
        module_name = import_map.get(type_name)
        if not module_name:
            continue
        append_type_context(type_name, module_name)

    # Transitive extraction (one level): for each extracted stub, find __init__ param types and extract their stubs
    # Build an extended import map that includes imports from source modules of already-extracted stubs
    transitive_import_map = dict(import_map)
    for _, cached_tree in module_cache.values():
        for cache_node in ast.walk(cached_tree):
            if isinstance(cache_node, ast.ImportFrom) and cache_node.module:
                for alias in cache_node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name not in transitive_import_map:
                        transitive_import_map[name] = cache_node.module

    emitted_names = type_names | existing_class_names | emitted_class_names | BUILTIN_AND_TYPING_NAMES
    transitive_type_names: set[str] = set()
    for cs in code_strings:
        try:
            stub_tree = ast.parse(cs.code)
        except SyntaxError:
            continue
        import_aliases = _collect_import_aliases(stub_tree)
        for stub_node in ast.walk(stub_tree):
            if isinstance(stub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and stub_node.name in (
                "__init__",
                "__post_init__",
            ):
                for arg in stub_node.args.args + stub_node.args.posonlyargs + stub_node.args.kwonlyargs:
                    transitive_type_names |= collect_type_names_from_annotation(arg.annotation)
            elif isinstance(stub_node, ast.ClassDef):
                transitive_type_names |= _collect_synthetic_constructor_type_names(stub_node, import_aliases)
    transitive_type_names -= emitted_names
    for type_name in sorted(transitive_type_names):
        module_name = transitive_import_map.get(type_name)
        if not module_name:
            continue
        append_type_context(type_name, module_name, transitive=True)

    return CodeStringsMarkdown(code_strings=code_strings)


def resolve_instance_class_name(name: str, module_tree: ast.Module) -> str | None:
    for node in module_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Call):
                        func = value.func
                        if isinstance(func, ast.Name):
                            return func.id
                        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            return func.value.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            ann = node.annotation
            if isinstance(ann, ast.Name):
                return ann.id
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                return ann.value.id
    return None


def enrich_testgen_context(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    import jedi

    result = _parse_and_collect_imports(code_context)
    if result is None:
        return CodeStringsMarkdown(code_strings=[])
    tree, imported_names = result

    if not imported_names:
        return CodeStringsMarkdown(code_strings=[])

    existing_classes = collect_existing_class_names(tree)

    code_strings: list[CodeString] = []
    emitted_class_names: set[str] = set()

    # --- Step 1: Project class definitions (jedi resolution + recursive base extraction) ---
    extracted_classes: set[tuple[Path, str]] = set()
    module_cache: dict[Path, tuple[str, ast.Module]] = {}

    def get_module_source_and_tree(module_path: Path) -> tuple[str, ast.Module] | None:
        if module_path in module_cache:
            return module_cache[module_path]
        try:
            module_source = module_path.read_text(encoding="utf-8")
            module_tree = ast.parse(module_source)
        except Exception:
            return None
        else:
            module_cache[module_path] = (module_source, module_tree)
            return module_source, module_tree

    def extract_class_and_bases(
        class_name: str, module_path: Path, module_source: str, module_tree: ast.Module
    ) -> None:
        if (module_path, class_name) in extracted_classes:
            return

        class_node = None
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            return

        for base in class_node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                continue

            if base_name and base_name not in existing_classes:
                extract_class_and_bases(base_name, module_path, module_source, module_tree)

        if (module_path, class_name) in extracted_classes:
            return

        lines = module_source.split("\n")
        start_line = class_node.lineno
        if class_node.decorator_list:
            start_line = min(d.lineno for d in class_node.decorator_list)
        class_source = "\n".join(lines[start_line - 1 : class_node.end_lineno])

        full_source = class_source

        code_strings.append(CodeString(code=full_source, file_path=module_path))
        extracted_classes.add((module_path, class_name))
        emitted_class_names.add(class_name)

    for name, module_name in imported_names.items():
        if name in existing_classes or module_name == "__future__":
            continue
        try:
            test_code = f"import {module_name}"
            script = jedi.Script(test_code, project=jedi.Project(path=project_root_path))
            completions = script.goto(1, len(test_code))

            if not completions:
                continue

            module_path = completions[0].module_path
            if not module_path:
                continue

            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = module_str.startswith(str(project_root_path.resolve()) + os.sep)
            is_third_party = "site-packages" in module_str
            if not is_project and not is_third_party:
                continue

            mod_result = get_module_source_and_tree(module_path)
            if mod_result is None:
                continue
            module_source, module_tree = mod_result

            if is_project:
                extract_class_and_bases(name, module_path, module_source, module_tree)
                if (module_path, name) not in extracted_classes:
                    resolved_class = resolve_instance_class_name(name, module_tree)
                    if resolved_class and resolved_class not in existing_classes:
                        extract_class_and_bases(resolved_class, module_path, module_source, module_tree)
            elif is_third_party:
                target_name = name
                if not any(isinstance(n, ast.ClassDef) and n.name == name for n in ast.walk(module_tree)):
                    resolved_class = resolve_instance_class_name(name, module_tree)
                    if resolved_class:
                        target_name = resolved_class
                if target_name not in emitted_class_names:
                    stub = extract_init_stub_from_class(target_name, module_source, module_tree)
                    if stub:
                        code_strings.append(CodeString(code=stub, file_path=module_path))
                        emitted_class_names.add(target_name)

        except Exception:
            logger.debug(f"Error extracting class definition for {name} from {module_name}")
            continue

    return CodeStringsMarkdown(code_strings=code_strings)


def resolve_classes_from_modules(candidates: set[tuple[str, str]]) -> list[tuple[type, str]]:
    """Import modules and resolve candidate (class_name, module_name) pairs to class objects."""
    import importlib
    import inspect

    resolved: list[tuple[type, str]] = []
    module_cache: dict[str, object] = {}

    for class_name, module_name in candidates:
        try:
            module = module_cache.get(module_name)
            if module is None:
                module = importlib.import_module(module_name)
                module_cache[module_name] = module

            cls = getattr(module, class_name, None)
            if cls is not None and inspect.isclass(cls):
                resolved.append((cls, class_name))
        except (ImportError, ModuleNotFoundError, AttributeError):
            logger.debug(f"Failed to import {module_name}.{class_name}")

    return resolved


MAX_TRANSITIVE_DEPTH = 5


def extract_classes_from_type_hint(hint: object) -> list[type]:
    """Recursively extract concrete class objects from a type annotation.

    Unwraps Optional, Union, List, Dict, Callable, Annotated, etc.
    Filters out builtins and typing module types.
    """
    import typing

    classes: list[type] = []
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None)

    if origin is not None and args:
        for arg in args:
            classes.extend(extract_classes_from_type_hint(arg))
    elif isinstance(hint, type):
        module = getattr(hint, "__module__", "")
        if module not in ("builtins", "typing", "typing_extensions", "types"):
            classes.append(hint)
    # Handle typing.Annotated on older Pythons where __origin__ may not be set
    if hasattr(typing, "get_args") and origin is None and args is None:
        try:
            inner_args = typing.get_args(hint)
            if inner_args:
                for arg in inner_args:
                    classes.extend(extract_classes_from_type_hint(arg))
        except Exception:
            pass

    return classes


def resolve_transitive_type_deps(cls: type) -> list[type]:
    """Find external classes referenced in cls.__init__ type annotations.

    Returns classes from site-packages that have a custom __init__.
    """
    import inspect
    import typing

    try:
        init_method = getattr(cls, "__init__")
        hints = typing.get_type_hints(init_method)
    except Exception:
        return []

    deps: list[type] = []
    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        for dep_cls in extract_classes_from_type_hint(hint):
            if dep_cls is cls:
                continue
            init_method = getattr(dep_cls, "__init__", None)
            if init_method is None or init_method is object.__init__:
                continue
            try:
                class_file = Path(inspect.getfile(dep_cls))
            except (OSError, TypeError):
                continue
            if not path_belongs_to_site_packages(class_file):
                continue
            deps.append(dep_cls)

    return deps


def extract_init_stub(cls: type, class_name: str, require_site_packages: bool = True) -> CodeString | None:
    """Extract a stub containing the class definition with only its __init__ method.

    Args:
        cls: The class object to extract __init__ from
        class_name: Name to use for the class in the stub
        require_site_packages: If True, only extract from site-packages. If False, include stdlib too.

    """
    import inspect
    import textwrap

    init_method = getattr(cls, "__init__", None)
    if init_method is None or init_method is object.__init__:
        return None

    try:
        class_file = Path(inspect.getfile(cls))
    except (OSError, TypeError):
        return None

    if require_site_packages and not path_belongs_to_site_packages(class_file):
        return None

    try:
        init_source = inspect.getsource(init_method)
        init_source = textwrap.dedent(init_source)
    except (OSError, TypeError):
        return None

    parts = class_file.parts
    if "site-packages" in parts:
        idx = parts.index("site-packages")
        class_file = Path(*parts[idx + 1 :])

    class_source = f"class {class_name}:\n" + textwrap.indent(init_source, "    ")
    return CodeString(code=class_source, file_path=class_file)


def _is_project_module_cached(module_name: str, project_root_path: Path, cache: dict[str, bool]) -> bool:
    cached = cache.get(module_name)
    if cached is not None:
        return cached
    is_project = _is_project_module(module_name, project_root_path)
    cache[module_name] = is_project
    return is_project


def is_project_path(module_path: Path | None, project_root_path: Path) -> bool:
    if module_path is None:
        return False
    # site-packages must be checked first because .venv/site-packages is under project root
    if path_belongs_to_site_packages(module_path):
        return False
    try:
        module_path.resolve().relative_to(project_root_path.resolve())
        return True
    except ValueError:
        return False


def _is_project_module(module_name: str, project_root_path: Path) -> bool:
    """Check if a module is part of the project (not external/stdlib)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    else:
        if spec is None or spec.origin is None:
            return False
        return is_project_path(Path(spec.origin), project_root_path)


def extract_imports_for_class(module_tree: ast.Module, class_node: ast.ClassDef, module_source: str) -> str:
    """Extract import statements needed for a class definition.

    This extracts imports for base classes, decorators, and type annotations.
    """
    needed_names: set[str] = set()

    # Get base class names
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            needed_names.add(base.id)
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # For things like abc.ABC, we need the module name
            needed_names.add(base.value.id)

    # Get decorator names (e.g., dataclass, field)
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name):
            needed_names.add(decorator.id)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                needed_names.add(decorator.func.id)
            elif isinstance(decorator.func, ast.Attribute) and isinstance(decorator.func.value, ast.Name):
                needed_names.add(decorator.func.value.id)

    # Get type annotation names from class body (for dataclass fields)
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and item.annotation:
            collect_names_from_annotation(item.annotation, needed_names)
        # Also check for field() calls which are common in dataclasses
        elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
            if isinstance(item.value.func, ast.Name):
                needed_names.add(item.value.func.id)

    # Find imports that provide these names
    import_lines: list[str] = []
    source_lines = module_source.split("\n")
    added_imports: set[int] = set()  # Track line numbers to avoid duplicates

    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break

    return "\n".join(import_lines)


def collect_names_from_annotation(node: ast.expr, names: set[str]) -> None:
    """Recursively collect type annotation names from an AST node."""
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Subscript):
        collect_names_from_annotation(node.value, names)
        collect_names_from_annotation(node.slice, names)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            collect_names_from_annotation(elt, names)
    elif isinstance(node, ast.BinOp):  # For Union types with | syntax
        collect_names_from_annotation(node.left, names)
        collect_names_from_annotation(node.right, names)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        names.add(node.value.id)


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    """Removes the docstring from an indented block if it exists."""
    if not isinstance(indented_block.body[0], cst.SimpleStatementLine):
        return indented_block
    first_stmt = indented_block.body[0].body[0]
    if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
        return indented_block.with_changes(body=indented_block.body[1:])
    return indented_block


def parse_code_and_prune_cst(
    code: str,
    code_context_type: CodeContextType,
    target_functions: set[str],
    helpers_of_helper_functions: set[str] = set(),  # noqa: B006
    remove_docstrings: bool = False,
) -> cst.Module:
    """Parse and filter the code CST, returning the pruned Module."""
    module = cst.parse_module(code)
    defs_with_usages = collect_top_level_defs_with_usages(module, target_functions | helpers_of_helper_functions)

    if code_context_type == CodeContextType.READ_WRITABLE:
        filtered_node, found_target = prune_cst(
            module, target_functions, defs_with_usages=defs_with_usages, keep_class_init=True
        )
    elif code_context_type == CodeContextType.READ_ONLY:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=False,
            include_dunder_methods=True,
        )
    elif code_context_type == CodeContextType.TESTGEN:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_dunder_methods=True,
            include_init_dunder=True,
        )
    elif code_context_type == CodeContextType.HASHING:
        filtered_node, found_target = prune_cst(
            module, target_functions, remove_docstrings=True, exclude_init_from_targets=True
        )
    else:
        raise ValueError(f"Unknown code_context_type: {code_context_type}")  # noqa: EM102

    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return filtered_node
    raise ValueError("Pruning produced no module")


def prune_cst(
    node: cst.CSTNode,
    target_functions: set[str],
    prefix: str = "",
    *,
    defs_with_usages: dict[str, UsageInfo] | None = None,
    helpers: set[str] | None = None,
    remove_docstrings: bool = False,
    include_target_in_output: bool = True,
    exclude_init_from_targets: bool = False,
    keep_class_init: bool = False,
    include_dunder_methods: bool = False,
    include_init_dunder: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    """Unified function to prune CST nodes based on various filtering criteria.

    Args:
        node: The CST node to filter
        target_functions: Set of qualified function names that are targets
        prefix: Current qualified name prefix (for class methods)
        defs_with_usages: Dict of definitions with usage info (for READ_WRITABLE mode)
        helpers: Set of helper function qualified names (for READ_ONLY/TESTGEN modes)
        remove_docstrings: Whether to remove docstrings from output
        include_target_in_output: Whether to include target functions in output
        exclude_init_from_targets: Whether to exclude __init__ from targets (HASHING mode)
        keep_class_init: Whether to keep __init__ methods in classes (READ_WRITABLE mode)
        include_dunder_methods: Whether to include dunder methods (READ_ONLY/TESTGEN modes)
        include_init_dunder: Whether to include __init__ in dunder methods

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # Check if it's a helper function (higher priority than target)
        if helpers and qualified_name in helpers:
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), True
            return node, True

        # Check if it's a target function
        if qualified_name in target_functions:
            # Handle exclude_init_from_targets for HASHING mode
            if exclude_init_from_targets and node.name.value == "__init__":
                return None, False

            if include_target_in_output:
                if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                    return node.with_changes(body=remove_docstring_from_body(node.body)), True
                return node, True
            return None, True

        # Handle class __init__ for READ_WRITABLE mode
        if keep_class_init and node.name.value == "__init__":
            return node, False

        # Handle dunder methods for READ_ONLY/TESTGEN modes
        if (
            include_dunder_methods
            and len(node.name.value) > 4
            and node.name.value.startswith("__")
            and node.name.value.endswith("__")
        ):
            if not include_init_dunder and node.name.value == "__init__":
                return None, False
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), False
            return node, False

        return None, False

    if isinstance(node, cst.ClassDef):
        if prefix:
            return None, False
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = node.name.value
        class_name = node.name.value

        # Handle dependency classes for READ_WRITABLE mode
        if defs_with_usages:
            # Check if this class contains any target functions
            has_target_functions = any(
                isinstance(stmt, cst.FunctionDef) and f"{class_prefix}.{stmt.name.value}" in target_functions
                for stmt in node.body.body
            )

            # If the class is used as a dependency (not containing target functions), keep it entirely
            if (
                not has_target_functions
                and class_name in defs_with_usages
                and defs_with_usages[class_name].used_by_qualified_function
            ):
                return node, True

        # Recursively filter each statement in the class body
        new_class_body: list[cst.CSTNode] = []
        found_in_class = False

        for stmt in node.body.body:
            filtered, found_target = prune_cst(
                stmt,
                target_functions,
                class_prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            )
            found_in_class |= found_target
            if filtered:
                new_class_body.append(filtered)

        if not found_in_class:
            return None, False

        # Apply docstring removal to class if needed
        if remove_docstrings and new_class_body:
            updated_body = node.body.with_changes(body=new_class_body)
            assert isinstance(updated_body, cst.IndentedBlock)
            return node.with_changes(body=remove_docstring_from_body(updated_body)), True

        return node.with_changes(body=node.body.with_changes(body=new_class_body)) if new_class_body else None, True

    # Handle assignments for READ_WRITABLE mode
    if defs_with_usages is not None:
        if isinstance(node, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
            if is_assignment_used(node, defs_with_usages):
                return node, True
            return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    if helpers is not None:
        return recurse_sections(
            node,
            section_names,
            lambda child: prune_cst(
                child,
                target_functions,
                prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            ),
            keep_non_target_children=True,
        )
    return recurse_sections(
        node,
        section_names,
        lambda child: prune_cst(
            child,
            target_functions,
            prefix,
            defs_with_usages=defs_with_usages,
            helpers=helpers,
            remove_docstrings=remove_docstrings,
            include_target_in_output=include_target_in_output,
            exclude_init_from_targets=exclude_init_from_targets,
            keep_class_init=keep_class_init,
            include_dunder_methods=include_dunder_methods,
            include_init_dunder=include_init_dunder,
        ),
    )


def belongs_to_method(name: Name, class_name: str, method_name: str) -> bool:
    """Check if the given name belongs to the specified method."""
    return belongs_to_function(name, method_name) and belongs_to_class(name, class_name)


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function."""
    if name.name == function_name:  # Handles function definition and recursive function calls
        return False
    if (name := name.parent()) and name.type == "function":
        return bool(name.name == function_name)
    return False


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if given jedi Name is a direct child of the specified class."""
    while name := name.parent():
        if name.type == "class":
            return bool(name.name == class_name)
    return False


def belongs_to_function_qualified(name: Name, qualified_function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function, matched by qualified function name."""
    try:
        if (
            name.full_name.startswith(name.module_name)
            and get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        ):
            # Handles function definition and recursive function calls
            return False
        if (name := name.parent()) and name.type == "function":
            return get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        return False
    except ValueError:
        return False
