import unittest
from unittest.mock import patch

from cli.codeflash.code_utils.git_utils import get_repo_owner_and_name


class TestGitUtils(unittest.TestCase):

    @patch('cli.codeflash.code_utils.git_utils.get_remote_url')
    def test_test_get_repo_owner_and_name(self, mock_get_remote_url):
        # Test with a standard GitHub HTTPS URL
        mock_get_remote_url.return_value = "https://github.com/owner/repo.git"
        owner, repo_name = get_repo_owner_and_name()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo_name, "repo")

        # Test with a GitHub SSH URL
        mock_get_remote_url.return_value = "git@github.com:owner/repo.git"
        owner, repo_name = get_repo_owner_and_name()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo_name, "repo")

        # Test with a URL without the .git suffix
        mock_get_remote_url.return_value = "https://github.com/owner/repo"
        owner, repo_name = get_repo_owner_and_name()
        self.assertEqual(owner, "owner")
        self.assertEqual(repo_name, "repo")


if __name__ == '__main__':
    unittest.main()
