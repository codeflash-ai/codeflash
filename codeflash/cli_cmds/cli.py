import logging
import os
from argparse import Namespace

import git

from codeflash.api.cfapi import check_github_app_installed_on_repo
from codeflash.cli_cmds.cmd_init import init_codeflash
from codeflash.code_utils import env_utils
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.git_utils import (
    git_root_dir,
    get_repo_owner_and_name,
    get_github_secrets_page_url,
)
from codeflash.version import __version__ as version

CF_BASE_URL = "https://app.codeflash.ai"


def process_cmd_args(args: Namespace) -> Namespace:
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    if args.version:
        logging.info(f"CodeFlash version {version}")
        exit()
    if "command" in args and args.command == "init":
        init_codeflash()
        exit()
    if args.function and not args.file:
        raise ValueError("If you specify a --function, you must specify the --file it is in")
    if args.file:
        if not os.path.exists(args.file):
            raise ValueError(f"File {args.file} does not exist")
        args.file = os.path.realpath(args.file)

    pyproject_config = parse_config_file(args.config_file)
    supported_keys = [
        "module_root",
        "tests_root",
        "test_framework",
        "ignore_paths",
        "minimum_performance_gain",
        "pytest_cmd",
    ]
    for key in supported_keys:
        if key in pyproject_config:
            if (
                hasattr(args, key.replace("-", "_"))
                and getattr(args, key.replace("-", "_")) is None
            ) or not hasattr(args, key.replace("-", "_")):
                setattr(args, key.replace("-", "_"), pyproject_config[key])
    assert os.path.isdir(
        args.module_root
    ), f"--module-root {args.module_root} must be a valid directory"
    assert os.path.isdir(
        args.tests_root
    ), f"--tests-root {args.tests_root} must be a valid directory"
    assert not (
        env_utils.get_pr_number() is not None and not env_utils.ensure_codeflash_api_key()
    ), (
        "CodeFlash API key not found. When running in a Github Actions Context, provide the "
        "'CODEFLASH_API_KEY' environment variable as a secret.\n"
        "You can add a secret by going to your repository's settings page, then clicking 'Secrets' in the left sidebar.\n"
        "Then, click 'New repository secret' and add your api key with the variable name CODEFLASH_API_KEY.\n"
        f"Here's a direct link: {get_github_secrets_page_url()}\n"
        "Exiting..."
    )
    if hasattr(args, "ignore_paths") and args.ignore_paths is not None:
        for path in args.ignore_paths:
            assert os.path.exists(
                path
            ), f"ignore-paths config must be a valid path. Path {path} does not exist"
    # Project root path is one level above the specified directory, because that's where the module can be imported from
    args.project_root = os.path.realpath(os.path.join(args.module_root, ".."))
    args.tests_root = os.path.realpath(args.tests_root)
    args = handle_optimize_all_arg_parsing(args)
    return args


def handle_optimize_all_arg_parsing(args: Namespace) -> Namespace:
    if hasattr(args, "all"):
        # Ensure that the user can actually open PRs on the repo.
        try:
            repo = git.Repo(search_parent_directories=True)
            git_root_dir(repo)
        except git.exc.InvalidGitRepositoryError:
            logging.error(
                "Could not find a git repository in the current directory. "
                "We need a git repository to run --all and open PRs for optimizations. Exiting..."
            )
            exit(1)
        owner, repo = get_repo_owner_and_name(repo)
        try:
            response = check_github_app_installed_on_repo(owner, repo)
            if response.ok and response.text == "true":
                pass
            else:
                logging.error(f"Error: {response.text}")
                raise Exception
        except Exception as e:
            logging.error(
                f"Could not find the CodeFlash GitHub App installed on the repository {owner}/{repo}. "
                "Please install the CodeFlash GitHub App on your repository to use --all."
                " Instructions at https://app.codeflash.ai \n"
                "Exiting..."
            )
            exit(1)
    if not hasattr(args, "all"):
        setattr(args, "all", None)
    elif args.all == "":
        # The default behavior of --all is to optimize everything in args.module_root
        args.all = args.module_root
    else:
        args.all = os.path.realpath(args.all)
    return args
