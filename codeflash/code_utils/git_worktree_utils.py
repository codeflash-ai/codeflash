from __future__ import annotations

import configparser
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import git
from git.exc import GitCommandError

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import codeflash_cache_dir
from codeflash.code_utils.git_utils import check_running_in_git_repo, git_root_dir

worktree_dirs = codeflash_cache_dir / "worktrees"
patches_dir = codeflash_cache_dir / "patches"


def create_worktree_snapshot_commit(worktree_dir: Path, commit_message: str) -> None:
    repository = git.Repo(worktree_dir, search_parent_directories=True)
    username = None
    no_username = False
    email = None
    no_email = False
    with repository.config_reader(config_level="repository") as cr:
        try:
            username = cr.get("user", "name")
        except (configparser.NoSectionError, configparser.NoOptionError):
            no_username = True
        try:
            email = cr.get("user", "email")
        except (configparser.NoSectionError, configparser.NoOptionError):
            no_email = True
    with repository.config_writer(config_level="repository") as cw:
        if not cw.has_option("user", "name"):
            cw.set_value("user", "name", "Codeflash Bot")
        if not cw.has_option("user", "email"):
            cw.set_value("user", "email", "bot@codeflash.ai")

    repository.git.add(".")
    repository.git.commit("-m", commit_message, "--no-verify")
    with repository.config_writer(config_level="repository") as cw:
        if username:
            cw.set_value("user", "name", username)
        elif no_username:
            cw.remove_option("user", "name")
        if email:
            cw.set_value("user", "email", email)
        elif no_email:
            cw.remove_option("user", "email")


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
        logger.info("!lsp|No uncommitted changes to copy to worktree.")
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


def _fallback_remove_worktree(worktree_dir: Path) -> None:
    """Fallback worktree removal using shutil.rmtree when git commands fail."""
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir, ignore_errors=True)
        logger.debug(f"Removed worktree directory using fallback method: {worktree_dir}")


def remove_worktree(worktree_dir: Path) -> None:
    """Remove a git worktree, with retry logic for Windows permission errors."""
    # Try to get repository reference
    try:
        repository = git.Repo(worktree_dir, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        # Worktree is not a valid git repository (corrupted or partially created)
        logger.debug(f"Worktree is not a valid git repository, using fallback deletion: {worktree_dir}")
        _fallback_remove_worktree(worktree_dir)
        return
    except Exception:
        logger.exception(f"Failed to open worktree repository: {worktree_dir}")
        _fallback_remove_worktree(worktree_dir)
        return

    # Try git worktree remove first
    for attempt in range(2):
        try:
            repository.git.worktree("remove", "--force", worktree_dir)
            logger.debug(f"Successfully removed worktree: {worktree_dir}")
            return
        except GitCommandError as e:
            error_msg = str(e).lower()
            # Check if it's a permission error or not a git repository error
            if "permission denied" in error_msg or "failed to delete" in error_msg:
                if attempt == 0:
                    # Retry once with a small delay to allow file handles to close
                    logger.debug(f"Permission error removing worktree (attempt {attempt + 1}), retrying after delay: {worktree_dir}")
                    time.sleep(0.5)
                    continue
            elif "not a git repository" in error_msg:
                # Worktree reference is broken, just delete the directory
                logger.debug(f"Worktree git reference is broken, using fallback deletion: {worktree_dir}")
                _fallback_remove_worktree(worktree_dir)
                return
            
            # Fallback to shutil.rmtree for any persistent error
            logger.warning(f"Git worktree remove failed, using fallback deletion: {worktree_dir}")
            _fallback_remove_worktree(worktree_dir)
            # Try to prune stale worktree references
            try:
                repository.git.worktree("prune")
            except Exception:
                pass
            return
        except Exception:
            logger.exception(f"Failed to remove worktree: {worktree_dir}")
            _fallback_remove_worktree(worktree_dir)
            return


def create_diff_patch_from_worktree(
    worktree_dir: Path, files: list[Path], fto_name: Optional[str] = None
) -> Optional[Path]:
    repository = git.Repo(worktree_dir, search_parent_directories=True)
    uni_diff_text = repository.git.diff(None, "HEAD", *files, ignore_blank_lines=True, ignore_space_at_eol=True)

    if not uni_diff_text:
        logger.warning("No changes found in worktree.")
        return None

    if not uni_diff_text.endswith("\n"):
        uni_diff_text += "\n"

    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = Path(patches_dir / f"{worktree_dir.name}.{fto_name}.patch")
    with patch_path.open("w", encoding="utf8") as f:
        f.write(uni_diff_text)

    return patch_path
