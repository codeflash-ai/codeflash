from __future__ import annotations

import configparser
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


def remove_worktree(worktree_dir: Path) -> None:
    """Remove a git worktree with robust error handling for Windows file locking issues.

    This function handles Windows-specific issues where files may be locked by processes,
    causing 'Permission denied' errors. It implements retry logic with exponential backoff
    and falls back to manual directory removal if git worktree remove fails.

    Args:
        worktree_dir: Path to the worktree directory to remove

    """
    if not worktree_dir or not worktree_dir.exists():
        logger.info(f"remove_worktree: Worktree does not exist, skipping removal. worktree_dir={worktree_dir}")
        return

    is_windows = sys.platform == "win32"
    max_retries = 3 if is_windows else 1
    retry_delay = 0.5  # Start with 500ms delay

    logger.info(
        f"remove_worktree: Starting worktree removal. worktree_dir={worktree_dir}, platform={sys.platform}, max_retries={max_retries}"
    )

    # Try to get the repository and git root for worktree removal
    try:
        repository = git.Repo(worktree_dir, search_parent_directories=True)
        git_root = repository.working_dir or repository.git_dir
        logger.info(f"remove_worktree: Found repository. git_root={git_root}")
    except Exception as e:
        logger.warning(
            f"remove_worktree: Could not access repository, attempting manual cleanup. worktree_dir={worktree_dir}, error={e}"
        )
        # If we can't access the repository, try manual cleanup
        _manual_cleanup_worktree_directory(worktree_dir)
        return

    # Attempt to remove worktree using git command with retries
    for attempt in range(max_retries):
        try:
            attempt_num = attempt + 1
            logger.info(
                f"remove_worktree: Attempting git worktree remove. attempt={attempt_num}, max_retries={max_retries}, worktree_dir={worktree_dir}"
            )
            repository.git.worktree("remove", "--force", str(worktree_dir))
            logger.info(f"remove_worktree: Successfully removed worktree via git command. worktree_dir={worktree_dir}")
            return  # noqa: TRY300
        except git.exc.GitCommandError as e:
            error_msg = str(e).lower()
            is_permission_error = "permission denied" in error_msg or "access is denied" in error_msg

            if is_permission_error and attempt < max_retries - 1:
                # On Windows, file locks may be temporary - retry with exponential backoff
                logger.info(
                    f"remove_worktree: Permission denied, retrying. attempt={attempt_num}, "
                    f"retry_delay={retry_delay}, worktree_dir={worktree_dir}, error={e}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Last attempt failed or non-permission error
                logger.warning(
                    f"remove_worktree: Git worktree remove failed, attempting fallback cleanup. "
                    f"attempts={attempt_num}, worktree_dir={worktree_dir}, error={e}"
                )
                break
        except Exception as e:
            logger.warning(
                f"remove_worktree: Unexpected error during git worktree remove, attempting fallback cleanup. "
                f"worktree_dir={worktree_dir}, error={e}"
            )
            break

    # Fallback: Try to remove worktree entry from git, then manually delete directory
    try:
        logger.info(f"remove_worktree: Attempting fallback cleanup. worktree_dir={worktree_dir}")

        # Try to prune the worktree entry from git (this doesn't delete the directory)
        try:
            # Use git worktree prune to remove stale entries
            repository.git.worktree("prune")
            logger.info("remove_worktree: Successfully pruned worktree entries")
        except Exception as prune_error:
            logger.info(f"remove_worktree: Could not prune worktree entries. error={prune_error}")

        # Manually remove the directory
        _manual_cleanup_worktree_directory(worktree_dir)

    except Exception as e:
        logger.error(
            f"remove_worktree: Failed to cleanup worktree directory after all attempts. "
            f"worktree_dir={worktree_dir}, error={e}"
        )


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
        logger.info(
            f"_manual_cleanup_worktree_directory: Directory does not exist, skipping. worktree_dir={worktree_dir}"
        )
        return

    # Validate paths for safety
    if not _validate_worktree_path_safety(worktree_dir):
        return

    logger.info(f"_manual_cleanup_worktree_directory: Starting manual directory removal. worktree_dir={worktree_dir}")

    # Attempt removal with retries on Windows
    is_windows = sys.platform == "win32"
    max_retries = 3 if is_windows else 1
    retry_delay = 0.5

    for attempt in range(max_retries):
        attempt_num = attempt + 1

        # Log retry attempts
        if attempt_num > 1:
            logger.info(
                f"_manual_cleanup_worktree_directory: Retrying directory removal. "
                f"attempt={attempt_num}, retry_delay={retry_delay}, worktree_dir={worktree_dir}"
            )
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
                logger.info(
                    f"_manual_cleanup_worktree_directory: Successfully removed directory. worktree_dir={worktree_dir}"
                )
                return

            # Directory still exists
            if attempt_num < max_retries:
                logger.info(
                    f"_manual_cleanup_worktree_directory: Directory still exists, will retry. "
                    f"attempt={attempt_num}, worktree_dir={worktree_dir}"
                )
            else:
                logger.warning(
                    f"_manual_cleanup_worktree_directory: Directory still exists after all attempts. "
                    f"attempts={attempt_num}, worktree_dir={worktree_dir}. "
                    f"Files may be locked and will be cleaned up later."
                )

        except Exception as e:
            if attempt_num < max_retries:
                logger.info(
                    f"_manual_cleanup_worktree_directory: Exception during removal, will retry. "
                    f"attempt={attempt_num}, error={e}"
                )
            else:
                logger.error(
                    f"_manual_cleanup_worktree_directory: Failed after all attempts. "
                    f"attempts={attempt_num}, worktree_dir={worktree_dir}, error={e}"
                )


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
    except (OSError, ValueError) as e:
        logger.error(
            f"_validate_worktree_path_safety: Failed to resolve paths, aborting for safety. "
            f"worktree_dir={worktree_dir}, error={e}"
        )
        return False

    # SAFETY CHECK 2: Ensure worktree_dir is a subdirectory of worktree_dirs
    try:
        # Use relative_to to check if path is under worktree_dirs
        worktree_dir_resolved.relative_to(worktree_dirs_resolved)
    except ValueError:
        logger.error(
            f"_validate_worktree_path_safety: Path is not under worktree_dirs, aborting for safety. "
            f"worktree_dir={worktree_dir_resolved}, worktree_dirs={worktree_dirs_resolved}"
        )
        return False

    # SAFETY CHECK 3: Ensure it's not the worktree_dirs root itself
    if worktree_dir_resolved == worktree_dirs_resolved:
        logger.error(
            f"_validate_worktree_path_safety: Attempted to delete worktree_dirs root, aborting for safety. "
            f"worktree_dir={worktree_dir_resolved}"
        )
        return False

    return True


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
        except Exception as e:
            # If it still fails, log and ignore (file is truly locked)
            logger.debug(
                f"_create_windows_rmtree_error_handler: Failed to remove file after permission change. path={path}, error={e}"
            )

    return handle_remove_error


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
