from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import git
from pygls import uris

from codeflash.api.cfapi import get_codeflash_api_key, get_user_id
from codeflash.cli_cmds.cli import process_pyproject_config
from codeflash.code_utils.git_worktree_utils import (
    create_diff_patch_from_worktree,
    get_patches_metadata,
    overwrite_patch_metadata,
)
from codeflash.code_utils.shell_utils import save_api_key_to_rc
from codeflash.discovery.functions_to_optimize import (
    filter_functions,
    get_functions_inside_a_commit,
    get_functions_within_git_diff,
)
from codeflash.either import is_successful
from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol

if TYPE_CHECKING:
    from argparse import Namespace

    from lsprotocol import types

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


@dataclass
class OptimizableFunctionsParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815


@dataclass
class FunctionOptimizationParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815
    functionName: str  # noqa: N815


@dataclass
class ProvideApiKeyParams:
    api_key: str


@dataclass
class OnPatchAppliedParams:
    patch_id: str


@dataclass
class OptimizableFunctionsInCommitParams:
    commit_hash: str


server = CodeflashLanguageServer("codeflash-language-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol)


@server.feature("getOptimizableFunctionsInCurrentDiff")
def get_functions_in_current_git_diff(
    server: CodeflashLanguageServer, _params: OptimizableFunctionsParams
) -> dict[str, str | dict[str, list[str]]]:
    functions = get_functions_within_git_diff(uncommitted_changes=True)
    file_to_qualified_names = _group_functions_by_file(server, functions)
    return {"functions": file_to_qualified_names, "status": "success"}


@server.feature("getOptimizableFunctionsInCommit")
def get_functions_in_commit(
    server: CodeflashLanguageServer, params: OptimizableFunctionsInCommitParams
) -> dict[str, str | dict[str, list[str]]]:
    functions = get_functions_inside_a_commit(params.commit_hash)
    file_to_qualified_names = _group_functions_by_file(server, functions)
    return {"functions": file_to_qualified_names, "status": "success"}


def _group_functions_by_file(
    server: CodeflashLanguageServer, functions: dict[str, list[FunctionToOptimize]]
) -> dict[str, list[str]]:
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
def get_optimizable_functions(
    server: CodeflashLanguageServer, params: OptimizableFunctionsParams
) -> dict[str, list[str]]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.show_message_log(f"Getting optimizable functions for: {file_path}", "Info")

    server.optimizer.args.file = file_path
    server.optimizer.args.function = None  # Always get ALL functions, not just one
    server.optimizer.args.previous_checkpoint_functions = False

    server.show_message_log(f"Calling get_optimizable_functions for {server.optimizer.args.file}...", "Info")
    optimizable_funcs, _, _ = server.optimizer.get_optimizable_functions()

    path_to_qualified_names = {}
    for functions in optimizable_funcs.values():
        path_to_qualified_names[file_path] = [func.qualified_name for func in functions]

    server.show_message_log(
        f"Found {len(path_to_qualified_names)} files with functions: {path_to_qualified_names}", "Info"
    )
    return path_to_qualified_names


@server.feature("initializeFunctionOptimization")
def initialize_function_optimization(
    server: CodeflashLanguageServer, params: FunctionOptimizationParams
) -> dict[str, str]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.show_message_log(f"Initializing optimization for function: {params.functionName} in {file_path}", "Info")

    if server.optimizer is None:
        _initialize_optimizer_if_api_key_is_valid(server)

    server.optimizer.worktree_mode()

    original_args, _ = server.optimizer.original_args_and_test_cfg

    server.optimizer.args.function = params.functionName
    original_relative_file_path = file_path.relative_to(original_args.project_root)
    server.optimizer.args.file = server.optimizer.current_worktree / original_relative_file_path
    server.optimizer.args.previous_checkpoint_functions = False

    server.show_message_log(
        f"Args set - function: {server.optimizer.args.function}, file: {server.optimizer.args.file}", "Info"
    )

    optimizable_funcs, count, _ = server.optimizer.get_optimizable_functions()

    if count == 0:
        server.show_message_log(f"No optimizable functions found for {params.functionName}", "Warning")
        server.cleanup_the_optimizer()
        return {"functionName": params.functionName, "status": "error", "message": "not found", "args": None}

    fto = optimizable_funcs.popitem()[1][0]
    server.optimizer.current_function_being_optimized = fto
    server.show_message_log(f"Successfully initialized optimization for {params.functionName}", "Info")
    return {"functionName": params.functionName, "status": "success"}


@server.feature("discoverFunctionTests")
def discover_function_tests(server: CodeflashLanguageServer, params: FunctionOptimizationParams) -> dict[str, str]:
    fto = server.optimizer.current_function_being_optimized
    optimizable_funcs = {fto.file_path: [fto]}

    devnull_writer = open(os.devnull, "w")  # noqa
    with contextlib.redirect_stdout(devnull_writer):
        function_to_tests, num_discovered_tests = server.optimizer.discover_tests(optimizable_funcs)

    server.optimizer.discovered_tests = function_to_tests

    return {"functionName": params.functionName, "status": "success", "discovered_tests": num_discovered_tests}


@server.feature("validateProject")
def validate_project(server: CodeflashLanguageServer, _params: FunctionOptimizationParams) -> dict[str, str]:
    from codeflash.cli_cmds.cmd_init import is_valid_pyproject_toml

    server.show_message_log("Validating project...", "Info")
    config = is_valid_pyproject_toml(server.args.config_file)
    if config is None:
        server.show_message_log("pyproject.toml is not valid", "Error")
        return {
            "status": "error",
            "message": "pyproject.toml is not valid",  # keep the error message the same, the extension is matching "pyproject.toml" in the error message to show the codeflash init instructions
        }

    args = process_args(server)
    repo = git.Repo(args.module_root, search_parent_directories=True)
    if repo.bare:
        return {"status": "error", "message": "Repository is in bare state"}

    try:
        _ = repo.head.commit
    except Exception:
        return {"status": "error", "message": "Repository has no commits (unborn HEAD)"}

    return {"status": "success"}


def _initialize_optimizer_if_api_key_is_valid(server: CodeflashLanguageServer) -> dict[str, str]:
    user_id = get_user_id()
    if user_id is None:
        return {"status": "error", "message": "api key not found or invalid"}

    if user_id.startswith("Error: "):
        error_msg = user_id[7:]
        return {"status": "error", "message": error_msg}

    from codeflash.optimization.optimizer import Optimizer

    new_args = process_args(server)
    server.optimizer = Optimizer(new_args)
    return {"status": "success", "user_id": user_id}


def process_args(server: CodeflashLanguageServer) -> Namespace:
    if server.args_processed_before:
        return server.args
    new_args = process_pyproject_config(server.args)
    server.args = new_args
    server.args_processed_before = True
    return new_args


@server.feature("apiKeyExistsAndValid")
def check_api_key(server: CodeflashLanguageServer, _params: any) -> dict[str, str]:
    try:
        return _initialize_optimizer_if_api_key_is_valid(server)
    except Exception:
        return {"status": "error", "message": "something went wrong while validating the api key"}


@server.feature("provideApiKey")
def provide_api_key(server: CodeflashLanguageServer, params: ProvideApiKeyParams) -> dict[str, str]:
    try:
        api_key = params.api_key
        if not api_key.startswith("cf-"):
            return {"status": "error", "message": "Api key is not valid"}

        result = save_api_key_to_rc(api_key)
        if not is_successful(result):
            return {"status": "error", "message": result.failure()}

        # clear cache to ensure the new api key is used
        get_codeflash_api_key.cache_clear()
        get_user_id.cache_clear()

        init_result = _initialize_optimizer_if_api_key_is_valid(server)
        if init_result["status"] == "error":
            return {"status": "error", "message": "Api key is not valid"}

        return {"status": "success", "message": "Api key saved successfully", "user_id": init_result["user_id"]}
    except Exception:
        return {"status": "error", "message": "something went wrong while saving the api key"}


@server.feature("retrieveSuccessfulOptimizations")
def retrieve_successful_optimizations(_server: CodeflashLanguageServer, _params: any) -> dict[str, str]:
    metadata = get_patches_metadata()
    return {"status": "success", "patches": metadata["patches"]}


@server.feature("onPatchApplied")
def on_patch_applied(_server: CodeflashLanguageServer, params: OnPatchAppliedParams) -> dict[str, str]:
    # first remove the patch from the metadata
    metadata = get_patches_metadata()

    deleted_patch_file = None
    new_patches = []
    for patch in metadata["patches"]:
        if patch["id"] == params.patch_id:
            deleted_patch_file = patch["patch_path"]
            continue
        new_patches.append(patch)

    # then remove the patch file
    if deleted_patch_file:
        overwrite_patch_metadata(new_patches)
        patch_path = Path(deleted_patch_file)
        patch_path.unlink(missing_ok=True)
        return {"status": "success"}
    return {"status": "error", "message": "Patch not found"}


@server.feature("performFunctionOptimization")
@server.thread()
def perform_function_optimization(  # noqa: PLR0911
    server: CodeflashLanguageServer, params: FunctionOptimizationParams
) -> dict[str, str]:
    try:
        server.show_message_log(f"Starting optimization for function: {params.functionName}", "Info")
        current_function = server.optimizer.current_function_being_optimized

        if not current_function:
            server.show_message_log(f"No current function being optimized for {params.functionName}", "Error")
            return {
                "functionName": params.functionName,
                "status": "error",
                "message": "No function currently being optimized",
            }

        module_prep_result = server.optimizer.prepare_module_for_optimization(current_function.file_path)

        validated_original_code, original_module_ast = module_prep_result

        function_optimizer = server.optimizer.create_function_optimizer(
            current_function,
            function_to_optimize_source_code=validated_original_code[current_function.file_path].source_code,
            original_module_ast=original_module_ast,
            original_module_path=current_function.file_path,
            function_to_tests=server.optimizer.discovered_tests or {},
        )

        server.optimizer.current_function_optimizer = function_optimizer
        if not function_optimizer:
            return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

        initialization_result = function_optimizer.can_be_optimized()
        if not is_successful(initialization_result):
            return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

        should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()

        test_setup_result = function_optimizer.generate_and_instrument_tests(
            code_context, should_run_experiment=should_run_experiment
        )
        if not is_successful(test_setup_result):
            return {"functionName": params.functionName, "status": "error", "message": test_setup_result.failure()}
        (
            generated_tests,
            function_to_concolic_tests,
            concolic_test_str,
            optimizations_set,
            generated_test_paths,
            generated_perf_test_paths,
            instrumented_unittests_created_for_function,
            original_conftest_content,
        ) = test_setup_result.unwrap()

        baseline_setup_result = function_optimizer.setup_and_establish_baseline(
            code_context=code_context,
            original_helper_code=original_helper_code,
            function_to_concolic_tests=function_to_concolic_tests,
            generated_test_paths=generated_test_paths,
            generated_perf_test_paths=generated_perf_test_paths,
            instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
            original_conftest_content=original_conftest_content,
        )

        if not is_successful(baseline_setup_result):
            return {"functionName": params.functionName, "status": "error", "message": baseline_setup_result.failure()}

        (
            function_to_optimize_qualified_name,
            function_to_all_tests,
            original_code_baseline,
            test_functions_to_remove,
            file_path_to_helper_classes,
        ) = baseline_setup_result.unwrap()

        best_optimization = function_optimizer.find_and_process_best_optimization(
            optimizations_set=optimizations_set,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            function_to_optimize_qualified_name=function_to_optimize_qualified_name,
            function_to_all_tests=function_to_all_tests,
            generated_tests=generated_tests,
            test_functions_to_remove=test_functions_to_remove,
            concolic_test_str=concolic_test_str,
        )

        if not best_optimization:
            server.show_message_log(
                f"No best optimizations found for function {function_to_optimize_qualified_name}", "Warning"
            )
            return {
                "functionName": params.functionName,
                "status": "error",
                "message": f"No best optimizations found for function {function_to_optimize_qualified_name}",
            }

        # generate a patch for the optimization
        relative_file_paths = [code_string.file_path for code_string in code_context.read_writable_code.code_strings]

        speedup = original_code_baseline.runtime / best_optimization.runtime

        # get the original file path in the actual project (not in the worktree)
        original_args, _ = server.optimizer.original_args_and_test_cfg
        relative_file_path = current_function.file_path.relative_to(server.optimizer.current_worktree)
        original_file_path = Path(original_args.project_root / relative_file_path).resolve()

        metadata = create_diff_patch_from_worktree(
            server.optimizer.current_worktree,
            relative_file_paths,
            metadata_input={
                "fto_name": function_to_optimize_qualified_name,
                "explanation": best_optimization.explanation_v2,
                "file_path": str(original_file_path),
                "speedup": speedup,
            },
        )

        server.show_message_log(f"Optimization completed for {params.functionName} with {speedup:.2f}x speedup", "Info")

        return {
            "functionName": params.functionName,
            "status": "success",
            "message": "Optimization completed successfully",
            "extra": f"Speedup: {speedup:.2f}x faster",
            "patch_file": metadata["patch_path"],
            "patch_id": metadata["id"],
            "explanation": best_optimization.explanation_v2,
        }
    finally:
        server.cleanup_the_optimizer()
