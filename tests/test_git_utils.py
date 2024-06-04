import unittest
from unittest.mock import patch

import git
from codeflash.code_utils.git_utils import (
    check_and_push_branch,
    check_running_in_git_repo,
    get_repo_owner_and_name,
)


class TestGitUtils(unittest.TestCase):
    @patch("codeflash.code_utils.git_utils.get_remote_url")
    def test_test_get_repo_owner_and_name(self, mock_get_remote_url):
        # Test with a standard GitHub HTTPS URL
        mock_get_remote_url.return_value = "https://github.com/owner/repo.git"
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

        # Test with a GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:owner/repo.git"
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

        # Test with another GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:codeflash-ai/posthog.git"
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "codeflash-ai"
        assert repo_name == "posthog"

        # Test with a URL without the .git suffix
        mock_get_remote_url.return_value = "https://github.com/owner/repo"
        owner, repo_name = get_repo_owner_and_name()
        assert owner == "owner"
        assert repo_name == "repo"

    @patch("codeflash.code_utils.git_utils.git.Repo")
    def test_check_running_in_git_repo_in_git_repo(self, mock_repo):
        mock_repo.return_value.git_dir = "/path/to/repo/.git"
        assert check_running_in_git_repo("/path/to/repo")

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=True)
    @patch("codeflash.code_utils.git_utils.confirm_proceeding_with_no_git_repo", return_value=True)
    def test_check_running_in_git_repo_not_in_git_repo_interactive(
        self,
        mock_confirm,
        mock_isatty,
        mock_repo,
    ):
        mock_repo.side_effect = git.exc.InvalidGitRepositoryError
        assert check_running_in_git_repo("/path/to/non-repo")

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=False)
    def test_check_running_in_git_repo_not_in_git_repo_non_interactive(self, mock_isatty, mock_repo):
        mock_repo.side_effect = git.exc.InvalidGitRepositoryError
        assert check_running_in_git_repo("/path/to/non-repo")

    @patch("codeflash.code_utils.git_utils.git.Repo")
    @patch("codeflash.code_utils.git_utils.sys.__stdin__.isatty", return_value=True)
    @patch("codeflash.code_utils.git_utils.inquirer.confirm", return_value=True)
    def test_check_and_push_branch(self, mock_confirm, mock_isatty, mock_repo):
        mock_repo_instance = mock_repo.return_value
        mock_repo_instance.active_branch.name = "test-branch"
        mock_repo_instance.refs = []

        mock_origin = mock_repo_instance.remote.return_value
        mock_origin.push.return_value = None

        assert check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_called_once_with("test-branch")
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
        mock_repo_instance.active_branch.name = "test-branch"
        mock_repo_instance.refs = []

        mock_origin = mock_repo_instance.remote.return_value
        mock_origin.push.return_value = None

        assert not check_and_push_branch(mock_repo_instance)
        mock_origin.push.assert_not_called()
        mock_origin.push.reset_mock()


if __name__ == "__main__":
    unittest.main()
