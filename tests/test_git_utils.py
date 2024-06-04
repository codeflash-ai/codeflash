import unittest
from unittest.mock import patch

import git
from codeflash.code_utils.git_utils import check_running_in_git_repo, get_repo_owner_and_name


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

    @patch("codeflash.code_utils.env_utils.git.Repo")
    def test_check_running_in_git_repo_in_git_repo(self, mock_repo):
        mock_repo.return_value.git_dir = "/path/to/repo/.git"
        assert check_running_in_git_repo("/path/to/repo")

    @patch("codeflash.code_utils.env_utils.git.Repo")
    @patch("codeflash.code_utils.env_utils.sys.__stdin__.isatty", return_value=True)
    @patch("codeflash.code_utils.env_utils.confirm_proceeding_with_no_git_repo", return_value=True)
    def test_check_running_in_git_repo_not_in_git_repo_interactive(
        self,
        mock_confirm,
        mock_isatty,
        mock_repo,
    ):
        mock_repo.side_effect = git.exc.InvalidGitRepositoryError
        assert check_running_in_git_repo("/path/to/non-repo")

    @patch("codeflash.code_utils.env_utils.git.Repo")
    @patch("codeflash.code_utils.env_utils.sys.__stdin__.isatty", return_value=False)
    def test_check_running_in_git_repo_not_in_git_repo_non_interactive(self, mock_isatty, mock_repo):
        mock_repo.side_effect = git.exc.InvalidGitRepositoryError
        assert check_running_in_git_repo("/path/to/non-repo")


if __name__ == "__main__":
    unittest.main()
