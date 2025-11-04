from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from codeflash.api.cfapi import get_codeflash_api_key, get_user_id
from codeflash.cli_cmds.cli import process_pyproject_config
from codeflash.cli_cmds.cmd_init import (
    CommonSections,
    VsCodeSetupInfo,
    configure_pyproject_toml,
    create_empty_pyproject_toml,
    get_formatter_cmds,
    get_suggestions,
    get_valid_subdirs,
    is_valid_pyproject_toml,
)
from codeflash.code_utils.git_utils import git_root_dir
from codeflash.code_utils.shell_utils import save_api_key_to_rc
from codeflash.discovery.functions_to_optimize import (
    filter_functions,
    get_functions_inside_a_commit,
    get_functions_within_git_diff,
)
from codeflash.either import is_successful
from codeflash.lsp.context import execution_context_vars
from codeflash.lsp.features.perform_optimization import get_cancelled_reponse, sync_perform_optimization
from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol

if TYPE_CHECKING:
    from argparse import Namespace

    from lsprotocol import types

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


@dataclass
class OptimizableFunctionsParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815


@dataclass
class FunctionOptimizationInitParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815
    functionName: str  # noqa: N815
    task_id: str


@dataclass
class FunctionOptimizationParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815
    functionName: str  # noqa: N815
    task_id: str


@dataclass
class ProvideApiKeyParams:
    api_key: str


@dataclass
class ValidateProjectParams:
    root_path_abs: str
    config_file: Optional[str] = None
    skip_validation: bool = False


@dataclass
class OptimizableFunctionsInCommitParams:
    commit_hash: str


@dataclass
class WriteConfigParams:
    config_file: str
    config: any


server = CodeflashLanguageServer("codeflash-language-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol)


@server.feature("server/listFeatures")
def list_features(_params: any) -> list[str]:
    return list(server.protocol.fm.features)


@server.feature("getOptimizableFunctionsInCurrentDiff")
def get_functions_in_current_git_diff(_params: OptimizableFunctionsParams) -> dict[str, str | dict[str, list[str]]]:
    functions = get_functions_within_git_diff(uncommitted_changes=True)
    file_to_qualified_names = _group_functions_by_file(functions)
    return {"functions": file_to_qualified_names, "status": "success"}


@server.feature("getOptimizableFunctionsInCommit")
def get_functions_in_commit(params: OptimizableFunctionsInCommitParams) -> dict[str, str | dict[str, list[str]]]:
    functions = get_functions_inside_a_commit(params.commit_hash)
    file_to_qualified_names = _group_functions_by_file(functions)
    return {"functions": file_to_qualified_names, "status": "success"}


def _group_functions_by_file(functions: dict[str, list[FunctionToOptimize]]) -> dict[str, list[str]]:
    file_to_funcs_to_optimize, _ = filter_functions(
        modified_functions=functions,
        tests_root=server.optimizer.test_cfg.tests_root,
        ignore_paths=[],
        project_root=server.optimizer.args.project_root,
        module_root=server.optimizer.args.module_root,
        previous_checkpoint_functions={},
    )
    file_to_qualified_names: dict[str, list[str]] = {
        str(path): [f.qualified_name for f in funcs] for path, funcs in file_to_funcs_to_optimize.items()
    }
    return file_to_qualified_names


@server.feature("getOptimizableFunctions")
def get_optimizable_functions(params: OptimizableFunctionsParams) -> dict[str, list[str]]:
    document_uri = params.textDocument.uri
    document = server.workspace.get_text_document(document_uri)

    file_path = Path(document.path)

    if not server.optimizer:
        return {"status": "error", "message": "optimizer not initialized"}

    server.optimizer.args.file = file_path
    server.optimizer.args.function = None  # Always get ALL functions, not just one
    server.optimizer.args.previous_checkpoint_functions = False

    optimizable_funcs, _, _ = server.optimizer.get_optimizable_functions()

    path_to_qualified_names = {}
    for functions in optimizable_funcs.values():
        path_to_qualified_names[file_path] = [func.qualified_name for func in functions]

    return path_to_qualified_names


def _find_pyproject_toml(workspace_path: str) -> tuple[Path | None, bool]:
    workspace_path_obj = Path(workspace_path)
    max_depth = 2
    base_depth = len(workspace_path_obj.parts)
    top_level_pyproject = None

    for root, dirs, files in os.walk(workspace_path_obj):
        depth = len(Path(root).parts) - base_depth
        if depth > max_depth:
            # stop going deeper into this branch
            dirs.clear()
            continue

        if "pyproject.toml" in files:
            file_path = Path(root) / "pyproject.toml"
            if depth == 0:
                top_level_pyproject = file_path
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip() == "[tool.codeflash]":
                        return file_path.resolve(), True
    return top_level_pyproject, False


@server.feature("writeConfig")
def write_config(params: WriteConfigParams) -> dict[str, any]:
    cfg = params.config
    cfg_file = Path(params.config_file) if params.config_file else None

    if cfg_file and not cfg_file.exists():
        # the client provided a config path but it doesn't exist
        create_empty_pyproject_toml(cfg_file)

    setup_info = VsCodeSetupInfo(
        module_root=getattr(cfg, "module_root", ""),
        tests_root=getattr(cfg, "tests_root", ""),
        test_framework=getattr(cfg, "test_framework", "pytest"),
        formatter=get_formatter_cmds(getattr(cfg, "formatter_cmds", "disabled")),
    )

    devnull_writer = open(os.devnull, "w")  # noqa
    with contextlib.redirect_stdout(devnull_writer):
        configured = configure_pyproject_toml(setup_info, cfg_file)
        if configured:
            return {"status": "success"}
        return {"status": "error", "message": "Failed to configure pyproject.toml"}


@server.feature("getConfigSuggestions")
def get_config_suggestions(_params: any) -> dict[str, any]:
    module_root_suggestions, default_module_root = get_suggestions(CommonSections.module_root)
    tests_root_suggestions, default_tests_root = get_suggestions(CommonSections.tests_root)
    test_framework_suggestions, default_test_framework = get_suggestions(CommonSections.test_framework)
    formatter_suggestions, default_formatter = get_suggestions(CommonSections.formatter_cmds)
    get_valid_subdirs.cache_clear()
    return {
        "module_root": {"choices": module_root_suggestions, "default": default_module_root},
        "tests_root": {"choices": tests_root_suggestions, "default": default_tests_root},
        "test_framework": {"choices": test_framework_suggestions, "default": default_test_framework},
        "formatter_cmds": {"choices": formatter_suggestions, "default": default_formatter},
    }


# should be called the first thing to initialize and validate the project
@server.feature("initProject")
def init_project(params: ValidateProjectParams) -> dict[str, str]:
    # Always process args in the init project, the extension can call
    server.initialized = False

    pyproject_toml_path: Path | None = getattr(params, "config_file", None) or getattr(server.args, "config_file", None)
    if pyproject_toml_path is not None:
        # if there is a config file provided use it
        server.prepare_optimizer_arguments(pyproject_toml_path)
    else:
        # otherwise look for it
        pyproject_toml_path, has_codeflash_config = _find_pyproject_toml(params.root_path_abs)
        if pyproject_toml_path and has_codeflash_config:
            server.show_message_log(f"Found pyproject.toml at: {pyproject_toml_path}", "Info")
            server.prepare_optimizer_arguments(pyproject_toml_path)
        elif pyproject_toml_path and not has_codeflash_config:
            return {
                "status": "error",
                "message": "pyproject.toml found in workspace, but no codeflash config.",
                "pyprojectPath": pyproject_toml_path,
            }
        else:
            return {"status": "error", "message": "No pyproject.toml found in workspace."}

    # since we are using worktrees, optimization diffs are generated with respect to the root of the repo.
    root = str(git_root_dir())

    if getattr(params, "skip_validation", False):
        return {
            "status": "success",
            "moduleRoot": server.args.module_root,
            "pyprojectPath": pyproject_toml_path,
            "root": root,
        }

    valid, config, reason = is_valid_pyproject_toml(pyproject_toml_path)
    if not valid:
        return {
            "status": "error",
            "message": f"reason: {reason}",
            "pyprojectPath": pyproject_toml_path,
            "existingConfig": config,
        }

    args = process_args()
    return {"status": "success", "moduleRoot": args.module_root, "pyprojectPath": pyproject_toml_path, "root": root}


def _initialize_optimizer_if_api_key_is_valid(api_key: Optional[str] = None) -> dict[str, str]:
    key_check_result = _check_api_key_validity(api_key)
    if key_check_result.get("status") != "success":
        return key_check_result

    _init()
    return key_check_result


def _check_api_key_validity(api_key: Optional[str]) -> dict[str, str]:
    user_id = get_user_id(api_key=api_key)
    if user_id is None:
        return {"status": "error", "message": "api key not found or invalid"}

    error_prefix = "Error: "
    if user_id.startswith(error_prefix):
        error_msg = user_id[len(error_prefix) :]
        return {"status": "error", "message": error_msg}

    return {"status": "success", "user_id": user_id}


def _initialize_optimizer(args: Namespace) -> None:
    from codeflash.optimization.optimizer import Optimizer

    if not server.optimizer:
        server.optimizer = Optimizer(args)


def process_args() -> Namespace:
    new_args = process_pyproject_config(server.args)
    server.args = new_args
    return new_args


def _init() -> Namespace:
    if server.initialized:
        return server.args
    new_args = process_args()
    _initialize_optimizer(new_args)
    server.initialized = True
    return new_args


@server.feature("apiKeyExistsAndValid")
def check_api_key(_params: any) -> dict[str, str]:
    try:
        return _initialize_optimizer_if_api_key_is_valid()
    except Exception:
        return {"status": "error", "message": "something went wrong while validating the api key"}


@server.feature("provideApiKey")
def provide_api_key(params: ProvideApiKeyParams) -> dict[str, str]:
    try:
        api_key = params.api_key
        if not api_key.startswith("cf-"):
            return {"status": "error", "message": "Api key is not valid"}

        # clear cache to ensure the new api key is used
        get_codeflash_api_key.cache_clear()
        get_user_id.cache_clear()

        key_check_result = _check_api_key_validity(api_key)
        if key_check_result.get("status") != "success":
            return key_check_result

        user_id = key_check_result["user_id"]
        result = save_api_key_to_rc(api_key)

        # initialize optimizer with the new api key
        _init()
        if not is_successful(result):
            return {"status": "error", "message": result.failure()}
        return {"status": "success", "message": "Api key saved successfully", "user_id": user_id}  # noqa: TRY300
    except Exception:
        return {"status": "error", "message": "something went wrong while saving the api key"}


@contextlib.contextmanager
def execution_context(**kwargs: str) -> None:
    """Temporarily set context values for the current async task."""
    # Create a fresh copy per use
    current = {**execution_context_vars.get(), **kwargs}
    token = execution_context_vars.set(current)
    try:
        yield
    finally:
        execution_context_vars.reset(token)


@server.feature("cleanupCurrentOptimizerSession")
def cleanup_optimizer(_params: any) -> dict[str, str]:
    if not server.cleanup_the_optimizer():
        return {"status": "error", "message": "Failed to cleanup optimizer"}
    return {"status": "success"}


@server.feature("initializeFunctionOptimization")
def initialize_function_optimization(params: FunctionOptimizationInitParams) -> dict[str, str]:
    with execution_context(task_id=getattr(params, "task_id", None)):
        document_uri = params.textDocument.uri
        document = server.workspace.get_text_document(document_uri)
        file_path = Path(document.path)

        server.show_message_log(
            f"Initializing optimization for function: {params.functionName} in {document_uri}", "Info"
        )

        if server.optimizer is None:
            _initialize_optimizer_if_api_key_is_valid()

        server.optimizer.args.file = file_path
        server.optimizer.args.function = params.functionName
        server.optimizer.args.previous_checkpoint_functions = False

        server.optimizer.worktree_mode()

        server.show_message_log(
            f"Args set - function: {server.optimizer.args.function}, file: {server.optimizer.args.file}", "Info"
        )

        optimizable_funcs, count, _ = server.optimizer.get_optimizable_functions()

        if count == 0:
            server.show_message_log(f"No optimizable functions found for {params.functionName}", "Warning")
            server.cleanup_the_optimizer()
            return {"functionName": params.functionName, "status": "error", "message": "not found", "args": None}

        fto = optimizable_funcs.popitem()[1][0]

        module_prep_result = server.optimizer.prepare_module_for_optimization(fto.file_path)
        if not module_prep_result:
            return {
                "functionName": params.functionName,
                "status": "error",
                "message": "Failed to prepare module for optimization",
            }

        validated_original_code, original_module_ast = module_prep_result

        function_optimizer = server.optimizer.create_function_optimizer(
            fto,
            function_to_optimize_source_code=validated_original_code[fto.file_path].source_code,
            original_module_ast=original_module_ast,
            original_module_path=fto.file_path,
            function_to_tests={},
        )

        server.optimizer.current_function_optimizer = function_optimizer
        if not function_optimizer:
            return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

        initialization_result = function_optimizer.can_be_optimized()
        if not is_successful(initialization_result):
            return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

        server.current_optimization_init_result = initialization_result.unwrap()
        server.show_message_log(f"Successfully initialized optimization for {params.functionName}", "Info")

        files = [function_optimizer.function_to_optimize.file_path]

        _, _, original_helpers = server.current_optimization_init_result
        files.extend([str(helper_path) for helper_path in original_helpers])

        return {"functionName": params.functionName, "status": "success", "files_inside_context": files}


@server.feature("performFunctionOptimization")
async def perform_function_optimization(params: FunctionOptimizationParams) -> dict[str, str]:
    with execution_context(task_id=getattr(params, "task_id", None)):
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        try:
            ctx = contextvars.copy_context()
            return await loop.run_in_executor(None, ctx.run, sync_perform_optimization, server, cancel_event, params)
        except asyncio.CancelledError:
            cancel_event.set()
            return get_cancelled_reponse()
        finally:
            server.cleanup_the_optimizer()
