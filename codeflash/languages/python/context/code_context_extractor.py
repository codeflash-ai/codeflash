from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
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
    from pathlib import Path

    from jedi.api.classes import Name

    from codeflash.languages.base import DependencyResolver
    from codeflash.languages.python.context.unused_definition_remover import UsageInfo


@dataclass
class FileContextCache:
    original_module: cst.Module
    cleaned_module: cst.Module
    fto_names: set[str]
    hoh_names: set[str]
    helper_functions: list[FunctionSource]
    file_path: Path
    relative_path: Path


@dataclass
class AllContextResults:
    read_writable: CodeStringsMarkdown
    read_only: CodeStringsMarkdown
    hashing: CodeStringsMarkdown
    testgen: CodeStringsMarkdown
    file_caches: list[FileContextCache] = field(default_factory=list, repr=False)


def build_testgen_context(
    testgen_base: CodeStringsMarkdown,
    project_root_path: Path,
    *,
    include_enrichment: bool = True,
    function_to_optimize: FunctionToOptimize | None = None,
) -> CodeStringsMarkdown:
    testgen_context = testgen_base

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

    # Extract all code contexts in a single pass (one CST parse per file)
    all_ctx = extract_all_contexts_from_files(helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path)

    final_read_writable_code = all_ctx.read_writable

    # Ensure the target file is first in the code blocks so the LLM knows which file to optimize
    target_relative = function_to_optimize.file_path.resolve().relative_to(project_root_path.resolve())
    target_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path == target_relative]
    other_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path != target_relative]
    if target_blocks:
        final_read_writable_code.code_strings = target_blocks + other_blocks

    read_only_code_markdown = all_ctx.read_only

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
        read_only_code_no_docstrings = re_extract_from_cache(
            all_ctx.file_caches, CodeContextType.READ_ONLY, project_root_path
        )
        read_only_context_code = read_only_code_no_docstrings.markdown
        if final_read_writable_tokens + encoded_tokens_len(read_only_context_code) > optim_token_limit:
            logger.debug("Code context has exceeded token limit, removing read-only code")
            read_only_context_code = ""

    # Progressive fallback for testgen context token limits
    testgen_context = build_testgen_context(
        all_ctx.testgen, project_root_path, function_to_optimize=function_to_optimize
    )

    if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
        logger.debug("Testgen context exceeded token limit, removing docstrings")
        testgen_base_no_docs = re_extract_from_cache(all_ctx.file_caches, CodeContextType.TESTGEN, project_root_path)
        testgen_context = build_testgen_context(
            testgen_base_no_docs, project_root_path, function_to_optimize=function_to_optimize
        )

        if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
            logger.debug("Testgen context still exceeded token limit, removing enrichment")
            testgen_context = build_testgen_context(testgen_base_no_docs, project_root_path, include_enrichment=False)

            if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
                raise ValueError(TESTGEN_LIMIT_ERROR)
    code_hash_context = all_ctx.hashing.markdown
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


def extract_all_contexts_from_files(
    helpers_of_fto: dict[Path, set[FunctionSource]],
    helpers_of_helpers: dict[Path, set[FunctionSource]],
    project_root_path: Path,
) -> AllContextResults:
    """Extract all 4 code context types from files in a single pass, parsing each file only once."""
    # Deduplicate: remove HoH entries that overlap with FTO (without mutating the caller's dict)
    hoh_deduped: dict[Path, set[FunctionSource]] = {}
    helpers_of_helpers_no_overlap: dict[Path, set[FunctionSource]] = {}
    for file_path, function_sources in helpers_of_helpers.items():
        if file_path in helpers_of_fto:
            hoh_deduped[file_path] = function_sources - helpers_of_fto[file_path]
        else:
            helpers_of_helpers_no_overlap[file_path] = function_sources

    rw = CodeStringsMarkdown()
    ro = CodeStringsMarkdown()
    hashing = CodeStringsMarkdown()
    testgen = CodeStringsMarkdown()
    file_caches: list[FileContextCache] = []

    # Process files containing FTO helpers (all 4 context types)
    for file_path, function_sources in helpers_of_fto.items():
        fto_names = {func.qualified_name for func in function_sources}
        hoh_funcs = hoh_deduped.get(file_path, set())
        hoh_names = {func.qualified_name for func in hoh_funcs}
        rw_helper_functions = list(function_sources)
        all_helper_functions = list(function_sources | hoh_funcs)

        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue

        try:
            original_module = cst.parse_module(original_code)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path} with libcst: {type(e).__name__}: {e}")
            continue

        try:
            relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        except ValueError:
            relative_path = file_path

        # Compute defs once for fto_names and reuse across remove + prune
        fto_defs = collect_top_level_defs_with_usages(original_module, fto_names)
        # Clean by fto_names only (for RW)
        rw_cleaned = remove_unused_definitions_by_function_names(original_module, fto_names, defs_with_usages=fto_defs)
        # Clean by all names (for RO/HASH/TESTGEN) — reuse rw_cleaned if no extra HoH names
        all_names = fto_names | hoh_names
        all_cleaned = (
            remove_unused_definitions_by_function_names(original_module, all_names) if hoh_names else rw_cleaned
        )

        # READ_WRITABLE
        try:
            rw_pruned = parse_code_and_prune_cst(
                rw_cleaned,
                CodeContextType.READ_WRITABLE,
                fto_names,
                set(),
                remove_docstrings=False,
                defs_with_usages=fto_defs,
            )
            if rw_pruned.code.strip():
                rw_code = add_needed_imports_from_module(
                    src_module_code=original_module,
                    dst_module_code=rw_pruned,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=rw_helper_functions,
                )
                rw.code_strings.append(CodeString(code=rw_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting read-writable code: {e}")

        # READ_ONLY
        try:
            ro_pruned = parse_code_and_prune_cst(
                all_cleaned, CodeContextType.READ_ONLY, fto_names, hoh_names, remove_docstrings=False
            )
            if ro_pruned.code.strip():
                ro_code = add_needed_imports_from_module(
                    src_module_code=original_module,
                    dst_module_code=ro_pruned,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=all_helper_functions,
                )
                ro.code_strings.append(CodeString(code=ro_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")

        # HASHING
        try:
            hash_pruned = parse_code_and_prune_cst(
                all_cleaned, CodeContextType.HASHING, fto_names, hoh_names, remove_docstrings=True
            )
            if hash_pruned.code.strip():
                hash_code = ast.unparse(ast.parse(hash_pruned.code))
                hashing.code_strings.append(CodeString(code=hash_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting hashing code: {e}")

        # TESTGEN
        try:
            testgen_pruned = parse_code_and_prune_cst(
                all_cleaned, CodeContextType.TESTGEN, fto_names, hoh_names, remove_docstrings=False
            )
            if testgen_pruned.code.strip():
                testgen_code = add_needed_imports_from_module(
                    src_module_code=original_module,
                    dst_module_code=testgen_pruned,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=all_helper_functions,
                )
                testgen.code_strings.append(CodeString(code=testgen_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting testgen code: {e}")

        file_caches.append(
            FileContextCache(
                original_module=original_module,
                cleaned_module=all_cleaned,
                fto_names=fto_names,
                hoh_names=hoh_names,
                helper_functions=all_helper_functions,
                file_path=file_path,
                relative_path=relative_path,
            )
        )

    # Process files containing only helpers of helpers (RO/HASH/TESTGEN only)
    for file_path, function_sources in helpers_of_helpers_no_overlap.items():
        hoh_names = {func.qualified_name for func in function_sources}
        helper_functions = list(function_sources)

        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue

        try:
            original_module = cst.parse_module(original_code)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path} with libcst: {type(e).__name__}: {e}")
            continue

        try:
            relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        except ValueError:
            relative_path = file_path

        cleaned = remove_unused_definitions_by_function_names(original_module, hoh_names)

        # READ_ONLY
        try:
            ro_pruned = parse_code_and_prune_cst(
                cleaned, CodeContextType.READ_ONLY, set(), hoh_names, remove_docstrings=False
            )
            if ro_pruned.code.strip():
                ro_code = add_needed_imports_from_module(
                    src_module_code=original_module,
                    dst_module_code=ro_pruned,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=helper_functions,
                )
                ro.code_strings.append(CodeString(code=ro_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")

        # HASHING
        try:
            hash_pruned = parse_code_and_prune_cst(
                cleaned, CodeContextType.HASHING, set(), hoh_names, remove_docstrings=True
            )
            if hash_pruned.code.strip():
                hash_code = ast.unparse(ast.parse(hash_pruned.code))
                hashing.code_strings.append(CodeString(code=hash_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting hashing code: {e}")

        # TESTGEN
        try:
            testgen_pruned = parse_code_and_prune_cst(
                cleaned, CodeContextType.TESTGEN, set(), hoh_names, remove_docstrings=False
            )
            if testgen_pruned.code.strip():
                testgen_code = add_needed_imports_from_module(
                    src_module_code=original_module,
                    dst_module_code=testgen_pruned,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=helper_functions,
                )
                testgen.code_strings.append(CodeString(code=testgen_code, file_path=relative_path))
        except ValueError as e:
            logger.debug(f"Error while getting testgen code: {e}")

        file_caches.append(
            FileContextCache(
                original_module=original_module,
                cleaned_module=cleaned,
                fto_names=set(),
                hoh_names=hoh_names,
                helper_functions=helper_functions,
                file_path=file_path,
                relative_path=relative_path,
            )
        )

    return AllContextResults(read_writable=rw, read_only=ro, hashing=hashing, testgen=testgen, file_caches=file_caches)


def re_extract_from_cache(
    file_caches: list[FileContextCache],
    code_context_type: CodeContextType,
    project_root_path: Path,
    remove_docstrings: bool = True,
) -> CodeStringsMarkdown:
    """Re-extract context from cached modules without file I/O or CST parsing."""
    result = CodeStringsMarkdown()
    for cache in file_caches:
        try:
            pruned = parse_code_and_prune_cst(
                cache.cleaned_module,
                code_context_type,
                cache.fto_names,
                cache.hoh_names,
                remove_docstrings=remove_docstrings,
            )
        except ValueError:
            continue
        if pruned.code.strip():
            code = add_needed_imports_from_module(
                src_module_code=cache.original_module,
                dst_module_code=pruned,
                src_path=cache.file_path,
                dst_path=cache.file_path,
                project_root=project_root_path,
                helper_functions=cache.helper_functions,
            )
            result.code_strings.append(CodeString(code=code, file_path=cache.relative_path))
    return result


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
                        if definition.type == "class":
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
    collector = ImportCollector()
    collector.visit(tree)
    return tree, collector.imported_names


def collect_existing_class_names(tree: ast.Module) -> set[str]:
    class_names = set()
    stack = [tree]

    while stack:
        node = stack.pop()

        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)

        # Only traverse nodes that can contain ClassDef nodes
        if isinstance(
            node,
            (
                ast.Module,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
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
            stack.extend(getattr(node, "body", []))
            stack.extend(getattr(node, "orelse", []))
            stack.extend(getattr(node, "finalbody", []))
            stack.extend(getattr(node, "handlers", []))

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
    return next((n for n in ast.walk(module_tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)


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
    for base in class_node.bases:  # noqa: SIM110
        if _expr_matches_name(base, import_aliases, "NamedTuple"):
            return True
    return False


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
    if class_node.decorator_list:
        return min(d.lineno for d in class_node.decorator_list)
    return class_node.lineno


def _class_has_explicit_init(class_node: ast.ClassDef) -> bool:
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            return True
    return False


def _collect_synthetic_constructor_type_names(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> set[str]:
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

        parameters.append(
            (item.target.id, _get_node_source(item.annotation, module_source, "Any"), default_value, kw_only)
        )
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
        class_node, module_source, import_aliases, kw_only_by_default=dataclass_kw_only
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


def _extract_function_stub_snippet(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, module_lines: list[str]) -> str:
    start_line = min(d.lineno for d in fn_node.decorator_list) if fn_node.decorator_list else fn_node.lineno
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
        if decorator_name and decorator_name.endswith((".setter", ".deleter")):
            continue
        return True
    return False


def _has_descriptor_like_class_fields(class_node: ast.ClassDef) -> bool:
    for item in class_node.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)) and isinstance(item.value, ast.Call):
            return True
    return False


def _should_use_raw_project_class_context(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> bool:
    start_line = _get_class_start_line(class_node)
    assert class_node.end_lineno is not None
    class_line_count = class_node.end_lineno - start_line + 1
    is_small = (
        class_line_count <= MAX_RAW_PROJECT_CLASS_LINES and len(class_node.body) <= MAX_RAW_PROJECT_CLASS_BODY_ITEMS
    )

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
    if not explicit_init_nodes:
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
            base_expr_name, module_tree, module_path, project_root_path, module_cache
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

    code_strings.append(
        CodeString(code=_extract_raw_class_context(class_node, module_source, module_tree), file_path=module_path)
    )
    emitted_classes.add(class_key)
    emitted_class_names.add(class_name)
    return True


def _collect_type_names_from_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module, class_name: str | None
) -> set[str]:
    type_names: set[str] = set()
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        type_names |= collect_type_names_from_annotation(arg.annotation)
    if func_node.args.vararg:
        type_names |= collect_type_names_from_annotation(func_node.args.vararg.annotation)
    if func_node.args.kwarg:
        type_names |= collect_type_names_from_annotation(func_node.args.kwarg.annotation)
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
            if (
                isinstance(body_node.left, ast.Call)
                and isinstance(body_node.left.func, ast.Name)
                and body_node.left.func.id == "type"
            ):
                for comparator in body_node.comparators:
                    if isinstance(comparator, ast.Name):
                        type_names.add(comparator.id)
    if class_name is not None:
        for top_node in ast.walk(tree):
            if isinstance(top_node, ast.ClassDef) and top_node.name == class_name:
                for base in top_node.bases:
                    if isinstance(base, ast.Name):
                        type_names.add(base.id)
                break
    return type_names


def _build_import_from_map(tree: ast.Module) -> dict[str, str]:
    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_map[alias.asname if alias.asname else alias.name] = node.module
    return import_map


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

    type_names = _collect_type_names_from_function(func_node, tree, function_to_optimize.class_name)
    type_names -= BUILTIN_AND_TYPING_NAMES
    type_names -= existing_class_names
    if not type_names:
        return CodeStringsMarkdown(code_strings=[])

    import_map = _build_import_from_map(tree)

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
    transitive_import_map = dict(import_map)
    for _, cached_tree in module_cache.values():
        for name, module in _build_import_from_map(cached_tree).items():
            transitive_import_map.setdefault(name, module)

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

    def extract_class_and_bases(
        class_name: str, module_path: Path, module_source: str, module_tree: ast.Module
    ) -> None:
        if (module_path, class_name) in extracted_classes:
            return

        class_node = _find_class_node_by_name(class_name, module_tree)
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
        class_source = "\n".join(lines[_get_class_start_line(class_node) - 1 : class_node.end_lineno])

        code_strings.append(CodeString(code=class_source, file_path=module_path))
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

            mod_result = _get_module_source_and_tree(module_path, module_cache)
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
                if _find_class_node_by_name(name, module_tree) is None:
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

    import_lines: list[str] = []
    source_lines = module_source.split("\n")
    added_imports: set[int] = set()
    for node in module_tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.lineno in added_imports:
            continue
        for alias in node.names:
            name = (
                alias.asname
                if alias.asname
                else (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
            )
            if name in needed_names:
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


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    if not isinstance(indented_block.body[0], cst.SimpleStatementLine):
        return indented_block
    first_stmt = indented_block.body[0].body[0]
    if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
        return indented_block.with_changes(body=indented_block.body[1:])
    return indented_block


def _maybe_strip_docstring(node: cst.FunctionDef | cst.ClassDef, cfg: PruneConfig) -> cst.FunctionDef | cst.ClassDef:
    if cfg.remove_docstrings and isinstance(node.body, cst.IndentedBlock):
        return node.with_changes(body=remove_docstring_from_body(node.body))
    return node


class ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imported_names: dict[str, str] = {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                if alias.name != "*":
                    self.imported_names[alias.asname if alias.asname else alias.name] = node.module


@dataclass(frozen=True)
class PruneConfig:
    defs_with_usages: dict[str, UsageInfo] | None = None
    helpers: set[str] | None = None
    remove_docstrings: bool = False
    include_target_in_output: bool = True
    exclude_init_from_targets: bool = False
    keep_class_init: bool = False
    include_dunder_methods: bool = False
    include_init_dunder: bool = False


def parse_code_and_prune_cst(
    code: str | cst.Module,
    code_context_type: CodeContextType,
    target_functions: set[str],
    helpers_of_helper_functions: set[str] = set(),  # noqa: B006
    remove_docstrings: bool = False,
    defs_with_usages: dict[str, UsageInfo] | None = None,
) -> cst.Module:
    """Parse and filter the code CST, returning the pruned Module."""
    module = code if isinstance(code, cst.Module) else cst.parse_module(code)

    if code_context_type == CodeContextType.READ_WRITABLE:
        if defs_with_usages is None:
            defs_with_usages = collect_top_level_defs_with_usages(
                module, target_functions | helpers_of_helper_functions
            )
        cfg = PruneConfig(defs_with_usages=defs_with_usages, keep_class_init=True)
    elif code_context_type == CodeContextType.READ_ONLY:
        cfg = PruneConfig(
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=False,
            include_dunder_methods=True,
        )
    elif code_context_type == CodeContextType.TESTGEN:
        cfg = PruneConfig(
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_dunder_methods=True,
            include_init_dunder=True,
        )
    elif code_context_type == CodeContextType.HASHING:
        cfg = PruneConfig(remove_docstrings=True, exclude_init_from_targets=True)
    else:
        raise ValueError(f"Unknown code_context_type: {code_context_type}")  # noqa: EM102

    filtered_node, found_target = prune_cst(module, target_functions, cfg)

    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return filtered_node
    raise ValueError("Pruning produced no module")


def prune_cst(
    node: cst.CSTNode, target_functions: set[str], cfg: PruneConfig, prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value

        if cfg.helpers and qualified_name in cfg.helpers:
            return _maybe_strip_docstring(node, cfg), True

        if qualified_name in target_functions:
            if cfg.exclude_init_from_targets and node.name.value == "__init__":
                return None, False
            if cfg.include_target_in_output:
                return _maybe_strip_docstring(node, cfg), True
            return None, True

        if cfg.keep_class_init and node.name.value == "__init__":
            return node, False

        if (
            cfg.include_dunder_methods
            and len(node.name.value) > 4
            and node.name.value.startswith("__")
            and node.name.value.endswith("__")
        ):
            if not cfg.include_init_dunder and node.name.value == "__init__":
                return None, False
            return _maybe_strip_docstring(node, cfg), False

        return None, False

    if isinstance(node, cst.ClassDef):
        if prefix:
            return None, False
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_name = node.name.value

        # Handle dependency classes for READ_WRITABLE mode
        if cfg.defs_with_usages:
            has_target_functions = any(
                isinstance(stmt, cst.FunctionDef) and f"{class_name}.{stmt.name.value}" in target_functions
                for stmt in node.body.body
            )
            if (
                not has_target_functions
                and class_name in cfg.defs_with_usages
                and cfg.defs_with_usages[class_name].used_by_qualified_function
            ):
                return node, True

        new_class_body: list[cst.CSTNode] = []
        found_in_class = False

        for stmt in node.body.body:
            filtered, found_target = prune_cst(stmt, target_functions, cfg, class_name)
            found_in_class |= found_target
            if filtered:
                new_class_body.append(filtered)

        if not found_in_class:
            return None, False
        if not new_class_body:
            return None, True
        updated = node.with_changes(body=node.body.with_changes(body=new_class_body))
        return _maybe_strip_docstring(updated, cfg), True

    # Handle assignments for READ_WRITABLE mode
    if cfg.defs_with_usages is not None:
        if isinstance(node, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
            if is_assignment_used(node, cfg.defs_with_usages):
                return node, True
            return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    return recurse_sections(
        node,
        section_names,
        lambda child: prune_cst(child, target_functions, cfg, prefix),
        keep_non_target_children=cfg.helpers is not None,
    )


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
