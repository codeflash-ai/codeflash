"""Safety tests for AiServiceClient.add_language_metadata().

These tests verify the correct payload structure for each language,
ensuring that merge resolution doesn't silently break the multi-language metadata logic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codeflash.api.aiservice import AiServiceClient
from codeflash.languages import Language


class TestAddLanguageMetadata:
    """Test add_language_metadata sets correct payload fields per language."""

    @patch("codeflash.api.aiservice.current_language", return_value=Language.PYTHON)
    def test_python_sets_language_version_and_python_version(self, _mock_lang: object) -> None:
        """For Python, both language_version and python_version should be set to the same value."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="3.11.5")
        assert payload["language_version"] == "3.11.5"
        assert payload["python_version"] == "3.11.5"
        assert "module_system" not in payload

    @patch("codeflash.api.aiservice.current_language", return_value=Language.PYTHON)
    def test_python_no_module_system(self, _mock_lang: object) -> None:
        """For Python, module_system should never be set even if provided."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="3.11.5", module_system="commonjs")
        assert "module_system" not in payload

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVA)
    def test_java_sets_language_version_not_python_version(self, _mock_lang: object) -> None:
        """For Java, language_version should be set, python_version should be None."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="17")
        assert payload["language_version"] == "17"
        assert payload["python_version"] is None

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVA)
    def test_java_includes_module_system(self, _mock_lang: object) -> None:
        """For Java, module_system should be set when provided."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="17", module_system="maven")
        assert payload["module_system"] == "maven"

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVA)
    def test_java_no_module_system_when_none(self, _mock_lang: object) -> None:
        """For Java, module_system should not be set when None."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="17", module_system=None)
        assert "module_system" not in payload

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVASCRIPT)
    def test_javascript_sets_language_version_not_python_version(self, _mock_lang: object) -> None:
        """For JavaScript, language_version should be set, python_version should be None."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="20.11.0")
        assert payload["language_version"] == "20.11.0"
        assert payload["python_version"] is None

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVASCRIPT)
    def test_javascript_includes_module_system(self, _mock_lang: object) -> None:
        """For JavaScript, module_system should be set when provided."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="20.11.0", module_system="esm")
        assert payload["module_system"] == "esm"

    @patch("codeflash.api.aiservice.current_language", return_value=Language.TYPESCRIPT)
    def test_typescript_same_as_javascript(self, _mock_lang: object) -> None:
        """TypeScript should behave the same as JavaScript."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version="20.11.0", module_system="commonjs")
        assert payload["language_version"] == "20.11.0"
        assert payload["python_version"] is None
        assert payload["module_system"] == "commonjs"

    @patch("codeflash.api.aiservice.current_language", return_value=Language.PYTHON)
    def test_none_language_version_python(self, _mock_lang: object) -> None:
        """When language_version is None for Python, payload should still have the keys."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version=None)
        assert payload["language_version"] is None
        assert payload["python_version"] is None

    @patch("codeflash.api.aiservice.current_language", return_value=Language.JAVA)
    def test_none_language_version_java(self, _mock_lang: object) -> None:
        """When language_version is None for Java, payload should still have the keys."""
        payload: dict = {}
        AiServiceClient.add_language_metadata(payload, language_version=None)
        assert payload["language_version"] is None
        assert payload["python_version"] is None
