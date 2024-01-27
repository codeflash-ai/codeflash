import git
import logging
import os
from argparse import Namespace

from codeflash.api.cfapi import check_github_app_installed_on_repo
from codeflash.code_utils.git_utils import git_root_dir, get_repo_owner_and_name
from codeflash.version import __version__ as version

CF_BASE_URL = "https://app.codeflash.ai"
LOGIN_URL = f"{CF_BASE_URL}/login"  # Replace with your actual URL
POLLING_URL = f"{CF_BASE_URL}/api/get-token"  # Replace with your actual polling endpoint
POLLING_INTERVAL = 10  # Polling interval in seconds
MAX_POLLING_ATTEMPTS = 30  # Maximum number of polling attempts

CODEFLASH_LOGO: str = (
    "\n"
    r"              __    _____         __ " + "\n"
    r" _______  ___/ /__ / _/ /__ ____ / / " + "\n"
    r"/ __/ _ \/ _  / -_) _/ / _ `(_-</ _ \ " + "\n"
    r"\__/\___/\_,_/\__/_//_/\_,_/___/_//_/" + "\n"
    f"{('v'+version).rjust(46)}\n"
    "                          https://codeflash.ai\n"
    "\n"
)


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
