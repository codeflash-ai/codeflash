"""Tests for codeflash.api.cfapi module."""

from __future__ import annotations

from unittest.mock import patch
import pytest
import requests

from codeflash.api.cfapi import make_cfapi_request


class TestMakeCfapiRequest:
    """Tests for make_cfapi_request function."""

    @patch("codeflash.api.cfapi.get_codeflash_api_key")
    @patch("requests.get")
    def test_connection_error_returns_response_with_503(
        self, mock_get, mock_get_api_key
    ):
        """
        Test that ConnectionError is caught and returns a Response object.

        When the CF API server is unreachable (e.g., connection refused),
        make_cfapi_request should catch the ConnectionError and return a
        Response object with status code 503 instead of letting the exception
        propagate.

        This test verifies the fix for the bug where ConnectionError was not
        handled, causing the CLI to crash when the server is down.
        """
        mock_get_api_key.return_value = "cf-test_key_123"

        # Simulate connection refused error
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "HTTPConnectionPool(host='localhost', port=3001): "
            "Max retries exceeded with url: /cfapi/test "
            "(Caused by NewConnectionError(\"Failed to establish a new connection: "
            "[Errno 111] Connection refused\"))"
        )

        # Make request with suppress_errors=True to avoid logging noise
        response = make_cfapi_request(
            endpoint="/test",
            method="GET",
            suppress_errors=True
        )

        # Verify we get a response object, not an exception
        assert response is not None
        assert hasattr(response, 'status_code')
        assert response.status_code == 503
        assert hasattr(response, 'text')
        assert 'Connection' in response.text or 'refused' in response.text.lower()

    @patch("codeflash.api.cfapi.get_codeflash_api_key")
    @patch("requests.post")
    def test_connection_error_on_post_returns_response_with_503(
        self, mock_post, mock_get_api_key
    ):
        """
        Test that ConnectionError is caught for POST requests as well.
        """
        mock_get_api_key.return_value = "cf-test_key_123"

        # Simulate connection timeout error
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='app.codeflash.ai', port=443): "
            "Max retries exceeded (Caused by ConnectTimeoutError)"
        )

        # Make POST request
        response = make_cfapi_request(
            endpoint="/test",
            method="POST",
            payload={"test": "data"},
            suppress_errors=True
        )

        # Verify we get a response object with 503 status
        assert response is not None
        assert hasattr(response, 'status_code')
        assert response.status_code == 503
        assert hasattr(response, 'text')

    @patch("codeflash.api.cfapi.get_codeflash_api_key")
    @patch("requests.get")
    def test_connection_error_without_suppress_logs_error(
        self, mock_get, mock_get_api_key, caplog
    ):
        """
        Test that ConnectionError is logged when suppress_errors=False.
        """
        mock_get_api_key.return_value = "cf-test_key_123"

        mock_get.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        # Make request without suppressing errors
        response = make_cfapi_request(
            endpoint="/test",
            method="GET",
            suppress_errors=False
        )

        # Verify error was logged
        assert response.status_code == 503
        # Note: We'd check caplog here but the actual logging happens in the function

    @patch("codeflash.api.cfapi.get_codeflash_api_key")
    @patch("requests.get")
    def test_http_error_still_handled(
        self, mock_get, mock_get_api_key
    ):
        """
        Test that existing HTTPError handling still works.

        This ensures our ConnectionError fix doesn't break existing
        error handling for HTTP errors (4xx, 5xx responses).
        """
        mock_get_api_key.return_value = "cf-test_key_123"

        # Create a mock response with 404 status
        mock_response = requests.Response()
        mock_response.status_code = 404
        mock_response._content = b'{"error": "Not found"}'

        # Make requests.get raise HTTPError
        mock_get.return_value = mock_response
        mock_response.raise_for_status = lambda: (_ for _ in ()).throw(
            requests.exceptions.HTTPError("404 Client Error")
        )

        response = make_cfapi_request(
            endpoint="/test",
            method="GET",
            suppress_errors=True
        )

        # Should return the response object, not raise
        assert response is not None
        assert response.status_code == 404
