from __future__ import annotations

import json
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git
from filelock import FileLock

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import codeflash_cache_dir
from codeflash.code_utils.git_utils import check_running_in_git_repo, git_root_dir

if TYPE_CHECKING:
    from typing import Any

    from git import Repo


worktree_dirs = codeflash_cache_dir / "worktrees"
patches_dir = codeflash_cache_dir / "patches"

if TYPE_CHECKING:
    from git import Repo


@lru_cache(maxsize=1)
def get_git_project_id() -> str:
    """Return the first commit sha of the repo."""
    repo: Repo = git.Repo(search_parent_directories=True)
    root_commits = list(repo.iter_commits(rev="HEAD", max_parents=0))
    return root_commits[0].hexsha


def create_worktree_snapshot_commit(worktree_dir: Path, commit_message: str) -> None:
    repository = git.Repo(worktree_dir, search_parent_directories=True)
    repository.git.add(".")
    repository.git.commit("-m", commit_message, "--no-verify")


def create_detached_worktree(module_root: Path) -> Optional[Path]:
    if not check_running_in_git_repo(module_root):
        logger.warning("Module is not in a git repository. Skipping worktree creation.")
        return None
    git_root = git_root_dir()
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    worktree_dir = worktree_dirs / f"{git_root.name}-{current_time_str}"

    repository = git.Repo(git_root, search_parent_directories=True)

    repository.git.worktree("add", "-d", str(worktree_dir))

    # Get uncommitted diff from the original repo
    repository.git.add("-N", ".")  # add the index for untracked files to be included in the diff
    exclude_binary_files = [":!*.pyc", ":!*.pyo", ":!*.pyd", ":!*.so", ":!*.dll", ":!*.whl", ":!*.egg", ":!*.egg-info", ":!*.pyz", ":!*.pkl", ":!*.pickle", ":!*.joblib", ":!*.npy", ":!*.npz", ":!*.h5", ":!*.hdf5", ":!*.pth", ":!*.pt", ":!*.pb", ":!*.onnx", ":!*.db", ":!*.sqlite", ":!*.sqlite3", ":!*.feather", ":!*.parquet", ":!*.jpg", ":!*.jpeg", ":!*.png", ":!*.gif", ":!*.bmp", ":!*.tiff", ":!*.webp", ":!*.wav", ":!*.mp3", ":!*.ogg", ":!*.flac", ":!*.mp4", ":!*.avi", ":!*.mov", ":!*.mkv", ":!*.pdf", ":!*.doc", ":!*.docx", ":!*.xls", ":!*.xlsx", ":!*.ppt", ":!*.pptx", ":!*.zip", ":!*.rar", ":!*.tar", ":!*.tar.gz", ":!*.tgz", ":!*.bz2", ":!*.xz"]  # fmt: off
    uni_diff_text = repository.git.diff(
        None, "HEAD", "--", *exclude_binary_files, ignore_blank_lines=True, ignore_space_at_eol=True
    )

    if not uni_diff_text.strip():
        logger.info("No uncommitted changes to copy to worktree.")
        return worktree_dir

    # Write the diff to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".codeflash.patch", delete=False) as tmp_patch_file:
        tmp_patch_file.write(uni_diff_text + "\n")  # the new line here is a must otherwise the last hunk won't be valid
        tmp_patch_file.flush()

        patch_path = Path(tmp_patch_file.name).resolve()

        # Apply the patch inside the worktree
        try:
            subprocess.run(
                ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", patch_path],
                cwd=worktree_dir,
                check=True,
            )
            create_worktree_snapshot_commit(worktree_dir, "Initial Snapshot")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch to worktree: {e}")

        return worktree_dir


def remove_worktree(worktree_dir: Path) -> None:
    try:
        repository = git.Repo(worktree_dir, search_parent_directories=True)
        repository.git.worktree("remove", "--force", worktree_dir)
    except Exception:
        logger.exception(f"Failed to remove worktree: {worktree_dir}")


@lru_cache(maxsize=1)
def get_patches_dir_for_project() -> Path:
    project_id = get_git_project_id() or ""
    return Path(patches_dir / project_id)


def get_patches_metadata() -> dict[str, Any]:
    project_patches_dir = get_patches_dir_for_project()
    meta_file = project_patches_dir / "metadata.json"
    if meta_file.exists():
        with meta_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"id": get_git_project_id() or "", "patches": []}


def save_patches_metadata(patch_metadata: dict) -> dict:
    project_patches_dir = get_patches_dir_for_project()
    meta_file = project_patches_dir / "metadata.json"
    lock_file = project_patches_dir / "metadata.json.lock"

    # we are not supporting multiple concurrent optimizations within the same process, but keep that in case we decide to do so in the future.
    with FileLock(lock_file, timeout=10):
        metadata = get_patches_metadata()

        patch_metadata["id"] = time.strftime("%Y%m%d-%H%M%S")
        metadata["patches"].append(patch_metadata)

        meta_file.write_text(json.dumps(metadata, indent=2))

    return patch_metadata


def overwrite_patch_metadata(patches: list[dict]) -> bool:
    project_patches_dir = get_patches_dir_for_project()
    meta_file = project_patches_dir / "metadata.json"
    lock_file = project_patches_dir / "metadata.json.lock"

    with FileLock(lock_file, timeout=10):
        metadata = get_patches_metadata()
        metadata["patches"] = patches
        meta_file.write_text(json.dumps(metadata, indent=2))
    return True


def create_diff_patch_from_worktree(
    worktree_dir: Path,
    files: list[str],
    fto_name: Optional[str] = None,
    metadata_input: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    repository = git.Repo(worktree_dir, search_parent_directories=True)
    uni_diff_text = repository.git.diff(None, "HEAD", *files, ignore_blank_lines=True, ignore_space_at_eol=True)

    if not uni_diff_text:
        logger.warning("No changes found in worktree.")
        return {}

    if not uni_diff_text.endswith("\n"):
        uni_diff_text += "\n"

    project_patches_dir = get_patches_dir_for_project()
    project_patches_dir.mkdir(parents=True, exist_ok=True)

    final_function_name = fto_name or metadata_input.get("fto_name", "unknown")
    patch_path = project_patches_dir / f"{worktree_dir.name}.{final_function_name}.patch"
    with patch_path.open("w", encoding="utf8") as f:
        f.write(uni_diff_text)

    final_metadata = {"patch_path": str(patch_path)}
    if metadata_input:
        final_metadata.update(metadata_input)
        final_metadata = save_patches_metadata(final_metadata)

    return final_metadata
