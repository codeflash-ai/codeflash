import unittest
from unittest.mock import patch

import git

from codeflash.code_utils.git_utils import (
    check_and_push_branch,
    check_running_in_git_repo,
    get_git_diff,
    get_repo_owner_and_name,
)


class TestGitUtils(unittest.TestCase):
    @patch("codeflash.code_utils.git_utils.get_remote_url")
    def test_test_get_repo_owner_and_name(self, mock_get_remote_url):
        # Test with a standard GitHub HTTPS URL
        mock_get_remote_url.return_value = "https://github.com/owner/repo.git"
        get_repo_owner_and_name.cache_clear()
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

        # Test with a GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:owner/repo.git"
        get_repo_owner_and_name.cache_clear()
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

        # Test with another GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:codeflash-ai/posthog.git"
        get_repo_owner_and_name.cache_clear()
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "codeflash-ai"
        assert repo_name == "posthog"

        # Test with a URL without the .git suffix
        mock_get_remote_url.return_value = "https://github.com/owner/repo"
        get_repo_owner_and_name.cache_clear()
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

        # Test with another GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:codeflash-ai/posthog/"
        get_repo_owner_and_name.cache_clear()
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "codeflash-ai"
        assert repo_name == "posthog"

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_check_running_in_git_repo_in_git_repo(self, mock_repo):
        mock_repo.return_value.git_dir = "/path/to/repo/.git"
        assert check_running_in_git_repo("/path/to/repo")

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=True)
    @patch("codeflash.code_utils.git_utils.confirm_proceeding_with_no_git_repo", return_value=True)
    def test_check_running_in_git_repo_not_in_git_repo_interactive(self, mock_confirm, mock_isatty, mock_repo):
        mock_repo.side_effect = git.InvalidGitRepositoryError  # type: ignore
        assert check_running_in_git_repo("/path/to/non-repo") == False

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=False)
    def test_check_running_in_git_repo_not_in_git_repo_non_interactive(self, mock_isatty, mock_repo):
        mock_repo.side_effect = git.exc.InvalidGitRepositoryError  # type: ignore
        assert check_running_in_git_repo("/path/to/non-repo") is False

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=True)
    @patch("codeflash.code_utils.git_utils.Confirm.ask", return_value=True)
    def test_check_and_push_branch(self, mock_confirm, mock_isatty, mock_repo):
        mock_repo_instance = mock_repo.return_value
        # Mock HEAD not being detached
        mock_repo_instance.head.is_detached = False
        mock_repo_instance.active_branch.name = "test-branch"
        mock_repo_instance.refs = []

        mock_origin = mock_repo_instance.remote.return_value
        mock_origin.push.return_value = None

        assert check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_called_once_with(mock_repo_instance.active_branch)
        mock_origin.push.reset_mock()

        # Test when branch is already pushed
        mock_repo_instance.refs = [f"origin/{mock_repo_instance.active_branch.name}"]
        assert check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_not_called()
        mock_origin.push.reset_mock()

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=False)
    def test_check_and_push_branch_non_tty(self, mock_isatty, mock_repo):
        mock_repo_instance = mock_repo.return_value
        # Mock HEAD not being detached
        mock_repo_instance.head.is_detached = False
        mock_repo_instance.active_branch.name = "test-branch"
        mock_repo_instance.refs = []

        mock_origin = mock_repo_instance.remote.return_value
        mock_origin.push.return_value = None

        assert not check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_not_called()
        mock_origin.push.reset_mock()

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_check_and_push_branch_detached_head(self, mock_repo):
        mock_repo_instance = mock_repo.return_value
        # Mock HEAD being detached
        mock_repo_instance.head.is_detached = True

        mock_origin = mock_repo_instance.remote.return_value
        mock_origin.push.return_value = None

        # Should return False when HEAD is detached
        assert not check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_not_called()


DELETION_ONLY_DIFF = """\
--- a/example.py
+++ b/example.py
@@ -5,7 +5,5 @@ def foo():
     a = 1
     b = 2
-    c = 3
-    d = 4
     e = 5
     return a + b + e

"""

ADDITION_ONLY_DIFF = """\
--- a/example.py
+++ b/example.py
@@ -5,5 +5,7 @@ def foo():
     a = 1
     b = 2
+    c = 3
+    d = 4
     e = 5
     return a + b + e

"""

MIXED_DIFF = """\
--- a/example.py
+++ b/example.py
@@ -5,6 +5,6 @@ def foo():
     a = 1
     b = 2
-    c = 3
+    c = 30
     e = 5
     return a + b + e

"""

MULTI_HUNK_DELETION_ONLY_DIFF = """\
--- a/example.py
+++ b/example.py
@@ -5,7 +5,5 @@ def foo():
     a = 1
     b = 2
-    c = 3
-    d = 4
     e = 5
     return a + b + e

@@ -20,6 +18,4 @@ def bar():
     x = 1
     y = 2
-    z = 3
-    w = 4
     return x + y

"""


class TestGetGitDiffDeletionOnly(unittest.TestCase):
    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_deletion_only_diff_returns_hunk_target_starts(self, mock_repo_cls):
        repo = mock_repo_cls.return_value
        repo.head.commit.hexsha = "abc123"
        repo.working_dir = "/repo"
        repo.git.diff.return_value = DELETION_ONLY_DIFF

        result = get_git_diff(repo_directory=None, uncommitted_changes=True)

        assert len(result) == 1
        key = list(result.keys())[0]
        assert str(key).endswith("example.py")
        # The hunk target_start is 5 — this is the fix: deletion-only diffs
        # should still report line numbers so the surrounding function is found.
        assert result[key] == [5]

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_addition_only_diff_returns_added_lines(self, mock_repo_cls):
        repo = mock_repo_cls.return_value
        repo.head.commit.hexsha = "abc123"
        repo.working_dir = "/repo"
        repo.git.diff.return_value = ADDITION_ONLY_DIFF

        result = get_git_diff(repo_directory=None, uncommitted_changes=True)

        key = list(result.keys())[0]
        # Added lines are at target line numbers 7 and 8
        assert result[key] == [7, 8]

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_mixed_diff_returns_only_added_lines(self, mock_repo_cls):
        repo = mock_repo_cls.return_value
        repo.head.commit.hexsha = "abc123"
        repo.working_dir = "/repo"
        repo.git.diff.return_value = MIXED_DIFF

        result = get_git_diff(repo_directory=None, uncommitted_changes=True)

        key = list(result.keys())[0]
        # Only the added line (c = 30) at target line 7
        assert result[key] == [7]

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_multi_hunk_deletion_only_returns_all_hunk_starts(self, mock_repo_cls):
        repo = mock_repo_cls.return_value
        repo.head.commit.hexsha = "abc123"
        repo.working_dir = "/repo"
        repo.git.diff.return_value = MULTI_HUNK_DELETION_ONLY_DIFF

        result = get_git_diff(repo_directory=None, uncommitted_changes=True)

        key = list(result.keys())[0]
        # Two hunks with target_start 5 and 18
        assert result[key] == [5, 18]

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_deletion_only_diff_does_not_return_empty_list(self, mock_repo_cls):
        repo = mock_repo_cls.return_value
        repo.head.commit.hexsha = "abc123"
        repo.working_dir = "/repo"
        repo.git.diff.return_value = DELETION_ONLY_DIFF

        result = get_git_diff(repo_directory=None, uncommitted_changes=True)

        key = list(result.keys())[0]
        # Without the fix, this would be an empty list, causing the function
        # to be missed during discovery.
        assert len(result[key]) > 0


if __name__ == "__main__":
    unittest.main()
