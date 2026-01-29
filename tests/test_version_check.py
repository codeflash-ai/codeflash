"""Tests for version checking functionality."""

import unittest
from unittest.mock import Mock, patch

from codeflash.code_utils.version_check import (
    _cache_duration,
    _version_cache,
    check_for_newer_minor_version,
    get_latest_version_from_pypi,
)


class TestVersionCheck(unittest.TestCase):
    """Test cases for version checking functionality."""

    def setUp(self):
        """Reset version cache before each test."""
        _version_cache["version"] = None
        _version_cache["timestamp"] = 0

    def tearDown(self):
        """Clean up after each test."""
        _version_cache["version"] = None
        _version_cache["timestamp"] = 0

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_success(self, mock_get):
        """Test successful version fetch from PyPI."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "1.2.3"}}
        mock_get.return_value = mock_response

        result = get_latest_version_from_pypi()

        self.assertEqual(result, "1.2.3")
        mock_get.assert_called_once_with("https://pypi.org/pypi/codeflash/json", timeout=2)

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_http_error(self, mock_get):
        """Test handling of HTTP error responses."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = get_latest_version_from_pypi()

        self.assertIsNone(result)

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_network_error(self, mock_get):
        """Test handling of network errors."""
        # Mock network error
        mock_get.side_effect = Exception("Network error")

        result = get_latest_version_from_pypi()

        self.assertIsNone(result)

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_invalid_response(self, mock_get):
        """Test handling of invalid response format."""
        # Mock invalid response format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}
        mock_get.return_value = mock_response

        result = get_latest_version_from_pypi()

        self.assertIsNone(result)

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_caching(self, mock_get):
        """Test that version caching works correctly."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "1.2.3"}}
        mock_get.return_value = mock_response

        # First call should hit the network
        result1 = get_latest_version_from_pypi()
        self.assertEqual(result1, "1.2.3")
        self.assertEqual(mock_get.call_count, 1)

        # Second call should use cache
        result2 = get_latest_version_from_pypi()
        self.assertEqual(result2, "1.2.3")
        self.assertEqual(mock_get.call_count, 1)  # Still only 1 call

    @patch("codeflash.code_utils.version_check.requests.get")
    def test_get_latest_version_from_pypi_cache_expiry(self, mock_get):
        """Test that cache expires after the specified duration."""
        import time

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "1.2.3"}}
        mock_get.return_value = mock_response

        # First call
        result1 = get_latest_version_from_pypi()
        self.assertEqual(result1, "1.2.3")

        # Manually expire the cache
        _version_cache["timestamp"] = time.time() - _cache_duration - 1

        # Second call should hit the network again
        result2 = get_latest_version_from_pypi()
        self.assertEqual(result2, "1.2.3")
        self.assertEqual(mock_get.call_count, 2)

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.0")
    def test_check_for_newer_minor_version_newer_available(self, mock_logger, mock_get_version):
        """Test warning message when newer minor version is available."""
        mock_get_version.return_value = "1.1.0"

        check_for_newer_minor_version()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("of Codeflash is available, please update soon!", call_args)
        self.assertIn("1.1.0", call_args)

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.0")
    def test_check_for_newer_minor_version_newer_major_available(self, mock_logger, mock_get_version):
        """Test warning message when newer major version is available."""
        mock_get_version.return_value = "2.0.0"

        check_for_newer_minor_version()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("of Codeflash is available, please update soon!", call_args)

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.1.0")
    def test_check_for_newer_minor_version_no_newer_available(self, mock_logger, mock_get_version):
        """Test no warning when no newer version is available."""
        mock_get_version.return_value = "1.0.0"

        check_for_newer_minor_version()

        mock_logger.warning.assert_not_called()

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.1")
    def test_check_for_newer_minor_version_patch_update_ignored(self, mock_logger, mock_get_version):
        """Test that patch updates don't trigger warnings."""
        mock_get_version.return_value = "1.0.1"

        check_for_newer_minor_version()

        mock_logger.warning.assert_not_called()

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.0")
    def test_check_for_newer_minor_version_same_version(self, mock_logger, mock_get_version):
        """Test no warning when versions are the same."""
        mock_get_version.return_value = "1.0.0"

        check_for_newer_minor_version()

        mock_logger.warning.assert_not_called()

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.0")
    def test_check_for_newer_minor_version_no_latest_version(self, mock_logger, mock_get_version):
        """Test no warning when latest version cannot be fetched."""
        mock_get_version.return_value = None

        check_for_newer_minor_version()

        mock_logger.warning.assert_not_called()

    @patch("codeflash.code_utils.version_check.get_latest_version_from_pypi")
    @patch("codeflash.code_utils.version_check.logger")
    @patch("codeflash.code_utils.version_check.__version__", "1.0.0")
    def test_check_for_newer_minor_version_invalid_version_format(self, mock_logger, mock_get_version):
        """Test handling of invalid version format."""
        mock_get_version.return_value = "invalid-version"

        check_for_newer_minor_version()

        mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
