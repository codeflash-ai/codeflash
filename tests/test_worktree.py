from argparse import Namespace
from pathlib import Path

import pytest
from codeflash.cli_cmds.cli import process_pyproject_config
from codeflash.optimization.optimizer import Optimizer


def test_mirror_paths_for_worktree_mode(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root / "code_to_optimize" / "code_directories" / "nested_module_root"

    monkeypatch.setattr("codeflash.optimization.optimizer.git_root_dir", lambda: project_root)

    args = Namespace()
    args.benchmark = False
    args.benchmarks_root = None
    args.no_pr = True

    args.config_file = project_root / "pyproject.toml"
    args.file = project_root / "src" / "app" / "main.py"
    args.worktree = True

    new_args = process_pyproject_config(args)

    optimizer = Optimizer(new_args)

    worktree_dir = repo_root / "worktree"
    optimizer.mirror_paths_for_worktree_mode(worktree_dir)

    assert optimizer.args.project_root == worktree_dir / "src"
    assert optimizer.args.test_project_root == worktree_dir / "src"
    assert optimizer.args.module_root == worktree_dir / "src" / "app"
    assert optimizer.args.tests_root == worktree_dir / "src" / "tests"
    assert optimizer.args.file == worktree_dir / "src" / "app" / "main.py"

    assert optimizer.test_cfg.tests_root == worktree_dir / "src" / "tests"
    assert optimizer.test_cfg.project_root_path == worktree_dir / "src" # same as project_root
    assert optimizer.test_cfg.tests_project_rootdir == worktree_dir / "src" # same as test_project_root

    # test on our repo
    monkeypatch.setattr("codeflash.optimization.optimizer.git_root_dir", lambda: repo_root)
    args = Namespace()
    args.benchmark = False
    args.benchmarks_root = None
    args.no_pr = True

    args.config_file = repo_root / "pyproject.toml"
    args.file = repo_root / "codeflash/optimization/optimizer.py"
    args.worktree = True

    new_args = process_pyproject_config(args)

    optimizer = Optimizer(new_args)

    worktree_dir = repo_root / "worktree"
    optimizer.mirror_paths_for_worktree_mode(worktree_dir)

    assert optimizer.args.project_root == worktree_dir
    assert optimizer.args.test_project_root == worktree_dir
    assert optimizer.args.module_root == worktree_dir / "codeflash"
    assert optimizer.args.tests_root == worktree_dir / "tests"
    assert optimizer.args.file == worktree_dir / "codeflash/optimization/optimizer.py"

    assert optimizer.test_cfg.tests_root == worktree_dir / "tests"
    assert optimizer.test_cfg.project_root_path == worktree_dir # same as project_root
    assert optimizer.test_cfg.tests_project_rootdir == worktree_dir # same as test_project_root
