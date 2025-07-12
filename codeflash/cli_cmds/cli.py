import logging
import sys
from argparse import SUPPRESS, ArgumentParser, Namespace
from pathlib import Path

from codeflash.cli_cmds import logging_config
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.cmd_init import init_codeflash, install_github_actions
from codeflash.cli_cmds.console import logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import exit_with_message
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.version import __version__ as version


def parse_args() -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    init_parser = subparsers.add_parser("init", help="Initialize Codeflash for a Python project.")
    init_parser.set_defaults(func=init_codeflash)

    init_actions_parser = subparsers.add_parser("init-actions", help="Initialize GitHub Actions workflow")
    init_actions_parser.set_defaults(func=install_github_actions)

    trace_optimize = subparsers.add_parser("optimize", help="Trace and optimize a Python project.")

    from codeflash.tracer import main as tracer_main

    trace_optimize.set_defaults(func=tracer_main)

    trace_optimize.add_argument(
        "--max-function-count",
        type=int,
        default=100,
        help="The maximum number of times to trace a single function. More calls to a function will not be traced. Default is 100.",
    )
    trace_optimize.add_argument(
        "--timeout",
        type=int,
        help="The maximum time in seconds to trace the entire workflow. Default is indefinite. This is useful while tracing really long workflows, to not wait indefinitely.",
    )
    trace_optimize.add_argument(
        "--output",
        type=str,
        default="codeflash.trace",
        help="The file to save the trace to. Default is codeflash.trace.",
    )
    trace_optimize.add_argument(
        "--config-file-path",
        type=str,
        help="The path to the pyproject.toml file which stores the Codeflash config. This is auto-discovered by default.",
    )

    parser.add_argument("--file", help="Try to optimize only this file")
    parser.add_argument("--function", help="Try to optimize only this function within the given file path")
    parser.add_argument(
        "--all",
        help="Try to optimize all functions. Can take a really long time. Can pass an optional starting directory to"
        " optimize code from. If no args specified (just --all), will optimize all code in the project.",
        nargs="?",
        const="",
        default=SUPPRESS,
    )
    parser.add_argument(
        "--module-root",
        type=str,
        help="Path to the project's Python module that you want to optimize."
        " This is the top-level root directory where all the Python source code is located.",
    )
    parser.add_argument(
        "--tests-root", type=str, help="Path to the test directory of the project, where all the tests are located."
    )
    parser.add_argument("--test-framework", choices=["pytest", "unittest"], default="pytest")
    parser.add_argument("--config-file", type=str, help="Path to the pyproject.toml with codeflash configs.")
    parser.add_argument("--replay-test", type=str, nargs="+", help="Paths to replay test to optimize functions from")
    parser.add_argument(
        "--no-pr", action="store_true", help="Do not create a PR for the optimization, only update the code locally."
    )
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Verify that codeflash is set up correctly by optimizing bubble sort as a test.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose debug logs")
    parser.add_argument("--version", action="store_true", help="Print the version of codeflash")
    parser.add_argument(
        "--benchmark", action="store_true", help="Trace benchmark tests and calculate optimization impact on benchmarks"
    )
    parser.add_argument(
        "--benchmarks-root",
        type=str,
        help="Path to the directory of the project, where all the pytest-benchmark tests are located.",
    )
    parser.add_argument("--no-draft", default=False, action="store_true", help="Skip optimization for draft PRs")

    args, unknown_args = parser.parse_known_args()
    sys.argv[:] = [sys.argv[0], *unknown_args]
    return process_and_validate_cmd_args(args)


def process_and_validate_cmd_args(args: Namespace) -> Namespace:
    from codeflash.code_utils.git_utils import (
        check_running_in_git_repo,
        confirm_proceeding_with_no_git_repo,
        get_repo_owner_and_name,
    )
    from codeflash.code_utils.github_utils import require_github_app_or_exit

    is_init: bool = args.command.startswith("init") if args.command else False
    if args.verbose:
        logging_config.set_level(logging.DEBUG, echo_setting=not is_init)
    else:
        logging_config.set_level(logging.INFO, echo_setting=not is_init)
    if args.version:
        logger.info(f"Codeflash version {version}")
        sys.exit()
    if not check_running_in_git_repo(module_root=args.module_root):
        if not confirm_proceeding_with_no_git_repo():
            exit_with_message("No git repository detected and user aborted run. Exiting...", error_on_exit=True)
        args.no_pr = True
    if args.function and not args.file:
        exit_with_message("If you specify a --function, you must specify the --file it is in", error_on_exit=True)
    if args.file:
        if not Path(args.file).exists():
            exit_with_message(f"File {args.file} does not exist", error_on_exit=True)
        args.file = Path(args.file).resolve()
        if not args.no_pr:
            owner, repo = get_repo_owner_and_name()
            require_github_app_or_exit(owner, repo)
    if args.replay_test:
        for test_path in args.replay_test:
            if not Path(test_path).is_file():
                exit_with_message(f"Replay test file {test_path} does not exist", error_on_exit=True)
        args.replay_test = [Path(replay_test).resolve() for replay_test in args.replay_test]
        if env_utils.is_ci():
            args.no_pr = True

    return args


def process_pyproject_config(args: Namespace) -> Namespace:
    try:
        pyproject_config, pyproject_file_path = parse_config_file(args.config_file)
    except ValueError as e:
        exit_with_message(f"Error parsing config file: {e}", error_on_exit=True)
    supported_keys = [
        "module_root",
        "tests_root",
        "benchmarks_root",
        "test_framework",
        "ignore_paths",
        "pytest_cmd",
        "formatter_cmds",
        "disable_telemetry",
        "disable_imports_sorting",
        "git_remote",
        "exp_git_remote",
        "override_fixtures",
    ]
    for key in supported_keys:
        if key in pyproject_config and (
            (hasattr(args, key.replace("-", "_")) and getattr(args, key.replace("-", "_")) is None)
            or not hasattr(args, key.replace("-", "_"))
        ):
            setattr(args, key.replace("-", "_"), pyproject_config[key])
    assert args.module_root is not None, "--module-root must be specified"
    assert Path(args.module_root).is_dir(), f"--module-root {args.module_root} must be a valid directory"
    assert args.tests_root is not None, "--tests-root must be specified"
    assert Path(args.tests_root).is_dir(), f"--tests-root {args.tests_root} must be a valid directory"
    if args.benchmark:
        assert args.benchmarks_root is not None, "--benchmarks-root must be specified when running with --benchmark"
        assert Path(args.benchmarks_root).is_dir(), (
            f"--benchmarks-root {args.benchmarks_root} must be a valid directory"
        )
        if env_utils.get_pr_number() is not None:
            import git

            from codeflash.code_utils.git_utils import get_repo_owner_and_name
            from codeflash.code_utils.github_utils import get_github_secrets_page_url, require_github_app_or_exit

            assert env_utils.ensure_codeflash_api_key(), (
                "Codeflash API key not found. When running in a Github Actions Context, provide the "
                "'CODEFLASH_API_KEY' environment variable as a secret.\n"
                "You can add a secret by going to your repository's settings page, then clicking 'Secrets' in the left sidebar.\n"
                "Then, click 'New repository secret' and add your api key with the variable name CODEFLASH_API_KEY.\n"
                f"Here's a direct link: {get_github_secrets_page_url()}\n"
                "Exiting..."
            )

            repo = git.Repo(search_parent_directories=True)

            owner, repo_name = get_repo_owner_and_name(repo)

            require_github_app_or_exit(owner, repo_name)

    if hasattr(args, "ignore_paths") and args.ignore_paths is not None:
        normalized_ignore_paths = []
        for path in args.ignore_paths:
            path_obj = Path(path)
            assert path_obj.exists(), f"ignore-paths config must be a valid path. Path {path} does not exist"
            normalized_ignore_paths.append(path_obj.resolve())
        args.ignore_paths = normalized_ignore_paths
    # Project root path is one level above the specified directory, because that's where the module can be imported from
    args.module_root = Path(args.module_root).resolve()
    # If module-root is "." then all imports are relatives to it.
    # in this case, the ".." becomes outside project scope, causing issues with un-importable paths
    args.project_root = project_root_from_module_root(args.module_root, pyproject_file_path)
    args.tests_root = Path(args.tests_root).resolve()
    if args.benchmarks_root:
        args.benchmarks_root = Path(args.benchmarks_root).resolve()
    args.test_project_root = project_root_from_module_root(args.tests_root, pyproject_file_path)
    return handle_optimize_all_arg_parsing(args)


def project_root_from_module_root(module_root: Path, pyproject_file_path: Path) -> Path:
    if pyproject_file_path.parent == module_root:
        return module_root
    return module_root.parent.resolve()


def handle_optimize_all_arg_parsing(args: Namespace) -> Namespace:
    if hasattr(args, "all"):
        import git

        from codeflash.code_utils.git_utils import check_and_push_branch, get_repo_owner_and_name
        from codeflash.code_utils.github_utils import require_github_app_or_exit

        # Ensure that the user can actually open PRs on the repo.
        try:
            git_repo = git.Repo(search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            logger.exception(
                "I couldn't find a git repository in the current directory. "
                "I need a git repository to run --all and open PRs for optimizations. Exiting..."
            )
            apologize_and_exit()
        if not args.no_pr and not check_and_push_branch(git_repo, git_remote=args.git_remote):
            exit_with_message("Branch is not pushed...", error_on_exit=True)
        owner, repo = get_repo_owner_and_name(git_repo)
        if not args.no_pr:
            require_github_app_or_exit(owner, repo)
    if not hasattr(args, "all"):
        args.all = None
    elif args.all == "":
        # The default behavior of --all is to optimize everything in args.module_root
        args.all = args.module_root
    else:
        args.all = Path(args.all).resolve()
    return args
