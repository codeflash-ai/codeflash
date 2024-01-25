import git
import logging
import os
from argparse import Namespace

from codeflash.code_utils.git_utils import git_root_dir
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
    if not hasattr(args, "all"):
        setattr(args, "all", None)
    if args.all is not None:
        try:
            git_root_dir()
        except git.exc.InvalidGitRepositoryError:
            logging.error(
                "Could not find a git repository in the current directory. "
                "We need a git repository to run --all and open PRs for optimizations. Exiting..."
            )
            exit(1)
    elif args.all == "":
        # The default behavior of --all is to optimize everything in args.module_root
        args.all = args.module_root
    else:
        args.all = os.path.realpath(args.all)
    return args
