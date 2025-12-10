from __future__ import annotations

import configparser
import contextlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Callable

import git

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
        except subprocess.CalledProcessError:
            pass

        return worktree_dir


def remove_worktree(worktree_dir: Path) -> None:
    """Remove a git worktree with robust error handling for Windows file locking issues.

    This function handles Windows-specific issues where files may be locked by processes,
    causing 'Permission denied' errors. It implements retry logic with exponential backoff
    and falls back to manual directory removal if git worktree remove fails.

    Args:
        worktree_dir: Path to the worktree directory to remove

    """
    if not worktree_dir or not worktree_dir.exists():
        return

    is_windows = sys.platform == "win32"
    max_retries = 3 if is_windows else 1
    retry_delay = 0.5  # Start with 500ms delay

    # Try to get the repository and git root for worktree removal
    try:
        repository = git.Repo(worktree_dir, search_parent_directories=True)
    except Exception:
        # If we can't access the repository, try manual cleanup
        _manual_cleanup_worktree_directory(worktree_dir)
        return

    # Attempt to remove worktree using git command with retries
    for attempt in range(max_retries):
        try:
            repository.git.worktree("remove", "--force", str(worktree_dir))
            return  # noqa: TRY300
        except git.exc.GitCommandError as e:
            error_msg = str(e).lower()
            is_permission_error = "permission denied" in error_msg or "access is denied" in error_msg

            if is_permission_error and attempt < max_retries - 1:
                # On Windows, file locks may be temporary - retry with exponential backoff
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Last attempt failed or non-permission error
                break
        except Exception:
            break

    # Fallback: Try to remove worktree entry from git, then manually delete directory
    with contextlib.suppress(Exception):
        # Try to prune the worktree entry from git (this doesn't delete the directory)
        # Use git worktree prune to remove stale entries
        repository.git.worktree("prune")

    # Manually remove the directory (always attempt, even if prune failed)
    with contextlib.suppress(Exception):
        _manual_cleanup_worktree_directory(worktree_dir)


def _manual_cleanup_worktree_directory(worktree_dir: Path) -> None:
    """Manually remove a worktree directory, handling Windows file locking issues.

    This is a fallback method when git worktree remove fails. It uses shutil.rmtree
    with custom error handling for Windows-specific issues.

    SAFETY: This function includes multiple safeguards to prevent accidental deletion:
    - Only deletes directories under the worktree_dirs cache location
    - Verifies the path is a worktree directory (not the original repo)
    - Uses resolve() to normalize paths and prevent path traversal attacks

    Args:
        worktree_dir: Path to the worktree directory to remove

    """
    if not worktree_dir or not worktree_dir.exists():
        return

    # Validate paths for safety
    if not _validate_worktree_path_safety(worktree_dir):
        return

    # Attempt removal with retries on Windows
    is_windows = sys.platform == "win32"
    max_retries = 3 if is_windows else 1
    retry_delay = 0.5

    for attempt in range(max_retries):
        attempt_num = attempt + 1

        if attempt_num > 1:
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

        # On Windows, use custom error handler to remove read-only attributes on the fly
        # This is more efficient than pre-scanning the entire directory tree
        error_handler = _create_windows_rmtree_error_handler() if is_windows else None

        try:
            if is_windows and error_handler:
                shutil.rmtree(worktree_dir, onerror=error_handler)
            else:
                shutil.rmtree(worktree_dir, ignore_errors=True)

            # Brief wait on Windows to allow file handles to be released
            if is_windows:
                wait_time = 0.3 if attempt_num < max_retries else 0.1
                time.sleep(wait_time)

            # Check if removal was successful
            if not worktree_dir.exists():
                return

        except Exception:  # noqa: S110
            pass


def _validate_worktree_path_safety(worktree_dir: Path) -> bool:
    """Validate that a path is safe to delete (must be under worktree_dirs).

    Args:
        worktree_dir: Path to validate

    Returns:
        True if the path is safe to delete, False otherwise

    """
    # SAFETY CHECK 1: Resolve paths to absolute, normalized paths
    try:
        worktree_dir_resolved = worktree_dir.resolve()
        worktree_dirs_resolved = worktree_dirs.resolve()
    except (OSError, ValueError):
        return False

    # SAFETY CHECK 2: Ensure worktree_dir is a subdirectory of worktree_dirs
    try:
        # Use relative_to to check if path is under worktree_dirs
        worktree_dir_resolved.relative_to(worktree_dirs_resolved)
    except ValueError:
        return False

    # SAFETY CHECK 3: Ensure it's not the worktree_dirs root itself
    return worktree_dir_resolved != worktree_dirs_resolved


def _create_windows_rmtree_error_handler() -> Callable[
    [Callable[[str], None], str, tuple[type[BaseException], BaseException, Any]], None
]:
    """Create an error handler for shutil.rmtree that handles Windows-specific issues.

    This handler attempts to remove read-only attributes when encountering permission errors.

    Returns:
        A callable error handler for shutil.rmtree's onerror parameter

    """

    def handle_remove_error(
        func: Callable[[str], None], path: str, exc_info: tuple[type[BaseException], BaseException, Any]
    ) -> None:
        """Error handler for shutil.rmtree on Windows.

        Attempts to remove read-only attributes and retry the operation.
        """
        # Get the exception type
        _exc_type, exc_value, _exc_traceback = exc_info

        # Only handle permission errors
        if not isinstance(exc_value, (PermissionError, OSError)):
            return

        try:
            # Try to change file permissions to make it writable
            # Using permissive mask (0o777) is intentional for Windows file cleanup
            Path(path).chmod(0o777)
            # Retry the failed operation
            func(path)
        except Exception:  # noqa: S110
            # If it still fails, silently ignore (file is truly locked)
            pass

    return handle_remove_error


def create_diff_patch_from_worktree(
    worktree_dir: Path, files: list[Path], fto_name: Optional[str] = None
) -> Optional[Path]:
    repository = git.Repo(worktree_dir, search_parent_directories=True)
    uni_diff_text = repository.git.diff(None, "HEAD", *files, ignore_blank_lines=True, ignore_space_at_eol=True)

    if not uni_diff_text:
        return None

    if not uni_diff_text.endswith("\n"):
        uni_diff_text += "\n"

    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = Path(patches_dir / f"{worktree_dir.name}.{fto_name}.patch")
    with patch_path.open("w", encoding="utf8") as f:
        f.write(uni_diff_text)

    return patch_path
