from pathlib import Path

from rich import print

from codeflash.cli_cmds.cli import process_pyproject_only
from codeflash.optimization.optimizer import Optimizer

args = process_pyproject_only(Path("/Users/krrt7/Desktop/work/codeflash/pyproject.toml"))


optimizer = Optimizer(args)
file_path = Path("/Users/krrt7/Desktop/work/codeflash/codeflash/code_utils/static_analysis.py")
optimizer.lsp_mode = True
optimizer.args.replay_test = None
optimizer.args.optimize_all = None
optimizer.args.file = file_path
optimizer.args.ignore_paths = []
optimizer.args.optimize_all = None
optimizer.args.replay_test = None
optimizer.args.only_get_this_function = None
optimizer.args.function = None
optimizer.args.benchmark = None

optimizer.discover_functions()
optimizer.discover_unit_tests()

