"""Extensive tests for the language registry module.

These tests verify that language registration, lookup, and detection
work correctly.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import Language
from codeflash.languages.registry import (
    UnsupportedLanguageError,
    clear_cache,
    clear_registry,
    detect_project_language,
    get_language_support,
    get_supported_extensions,
    get_supported_languages,
    is_language_supported,
    register_language,
)


@pytest.fixture(autouse=True)
def setup_registry():
    """Ensure PythonSupport is registered before each test."""
    # Import to trigger registration

    yield
    # Clear cache after each test to avoid side effects
    clear_cache()


class TestRegisterLanguage:
    """Tests for the register_language decorator."""

    def test_register_language_decorator(self):
        """Test that register_language decorator registers correctly."""
        # Python should already be registered via the fixture
        assert ".py" in get_supported_extensions()
        assert "python" in get_supported_languages()

    def test_registered_language_lookup_by_extension(self):
        """Test looking up registered language by extension."""
        support = get_language_support(".py")
        assert support.language == Language.PYTHON

    def test_registered_language_lookup_by_language(self):
        """Test looking up registered language by Language enum."""
        support = get_language_support(Language.PYTHON)
        assert support.language == Language.PYTHON


class TestGetLanguageSupport:
    """Tests for the get_language_support function."""

    def test_get_by_path_python(self):
        """Test getting language support by Python file path."""
        support = get_language_support(Path("/test/example.py"))
        assert support.language == Language.PYTHON

    def test_get_by_path_pyw(self):
        """Test getting language support by .pyw extension."""
        support = get_language_support(Path("/test/example.pyw"))
        assert support.language == Language.PYTHON

    def test_get_by_language_enum(self):
        """Test getting language support by Language enum."""
        support = get_language_support(Language.PYTHON)
        assert support.language == Language.PYTHON

    def test_get_by_extension_string(self):
        """Test getting language support by extension string."""
        support = get_language_support(".py")
        assert support.language == Language.PYTHON

    def test_get_by_extension_without_dot(self):
        """Test getting language support by extension without dot."""
        support = get_language_support("py")
        assert support.language == Language.PYTHON

    def test_get_by_language_name_string(self):
        """Test getting language support by language name string."""
        support = get_language_support("python")
        assert support.language == Language.PYTHON

    def test_unsupported_extension_raises(self):
        """Test that unsupported extension raises UnsupportedLanguageError."""
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            get_language_support(Path("/test/example.xyz"))
        assert "xyz" in str(exc_info.value.identifier) or "example.xyz" in str(exc_info.value.identifier)

    def test_unsupported_language_raises(self):
        """Test that unsupported language name raises UnsupportedLanguageError."""
        with pytest.raises(UnsupportedLanguageError):
            get_language_support("unknown_language")

    def test_caching(self):
        """Test that language support instances are cached."""
        support1 = get_language_support(Language.PYTHON)
        support2 = get_language_support(Language.PYTHON)
        assert support1 is support2

    def test_cache_cleared(self):
        """Test that cache can be cleared."""
        support1 = get_language_support(Language.PYTHON)
        clear_cache()
        support2 = get_language_support(Language.PYTHON)
        # After clearing cache, should be different instances
        assert support1 is not support2

    def test_case_insensitive_extension(self):
        """Test that extension lookup is case insensitive."""
        support1 = get_language_support(".PY")
        support2 = get_language_support(".py")
        assert support1.language == support2.language

    def test_case_insensitive_language_name(self):
        """Test that language name lookup is case insensitive."""
        support1 = get_language_support("PYTHON")
        support2 = get_language_support("python")
        assert support1.language == support2.language


class TestDetectProjectLanguage:
    """Tests for the detect_project_language function."""

    def test_detect_python_project(self):
        """Test detecting a Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create some Python files
            (tmpdir_path / "main.py").write_text("print('hello')")
            (tmpdir_path / "utils.py").write_text("def helper(): pass")
            (tmpdir_path / "subdir").mkdir()
            (tmpdir_path / "subdir" / "module.py").write_text("x = 1")

            language = detect_project_language(tmpdir_path, tmpdir_path)
            assert language == Language.PYTHON

    def test_detect_mixed_project_prefers_most_common(self):
        """Test that detection prefers the most common supported language."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create more Python files than other files
            for i in range(5):
                (tmpdir_path / f"module_{i}.py").write_text(f"x = {i}")

            # Create some unsupported files
            (tmpdir_path / "data.json").write_text("{}")
            (tmpdir_path / "readme.md").write_text("# Readme")

            language = detect_project_language(tmpdir_path, tmpdir_path)
            assert language == Language.PYTHON

    def test_detect_no_supported_language_raises(self):
        """Test that empty project raises UnsupportedLanguageError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create only unsupported files
            (tmpdir_path / "data.json").write_text("{}")
            (tmpdir_path / "readme.md").write_text("# Readme")

            with pytest.raises(UnsupportedLanguageError):
                detect_project_language(tmpdir_path, tmpdir_path)

    def test_detect_empty_project_raises(self):
        """Test that empty project raises UnsupportedLanguageError."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(UnsupportedLanguageError):
            detect_project_language(Path(tmpdir), Path(tmpdir))

    def test_detect_with_different_roots(self):
        """Test detection with different project and module roots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            module_root = tmpdir_path / "src"
            module_root.mkdir()

            # Create Python files only in module root
            (module_root / "main.py").write_text("print('hello')")

            # Root has no Python files
            (tmpdir_path / "config.json").write_text("{}")

            language = detect_project_language(tmpdir_path, module_root)
            assert language == Language.PYTHON


class TestSupportedLanguagesAndExtensions:
    """Tests for get_supported_languages and get_supported_extensions."""

    def test_get_supported_languages_includes_python(self):
        """Test that Python is in supported languages."""
        languages = get_supported_languages()
        assert "python" in languages

    def test_get_supported_extensions_includes_py(self):
        """Test that .py is in supported extensions."""
        extensions = get_supported_extensions()
        assert ".py" in extensions


class TestIsLanguageSupported:
    """Tests for the is_language_supported function."""

    def test_python_is_supported(self):
        """Test that Python is supported."""
        assert is_language_supported(Language.PYTHON) is True
        assert is_language_supported(".py") is True
        assert is_language_supported("python") is True
        assert is_language_supported(Path("/test/example.py")) is True

    def test_unknown_is_not_supported(self):
        """Test that unknown languages are not supported."""
        assert is_language_supported(".xyz") is False
        assert is_language_supported("unknown") is False
        assert is_language_supported(Path("/test/example.xyz")) is False


class TestUnsupportedLanguageError:
    """Tests for the UnsupportedLanguageError exception."""

    def test_error_message_includes_identifier(self):
        """Test that error message includes the identifier."""
        error = UnsupportedLanguageError(".xyz")
        assert ".xyz" in str(error)

    def test_error_message_includes_supported(self):
        """Test that error message includes supported languages."""
        error = UnsupportedLanguageError(".xyz", supported=["python", "javascript"])
        msg = str(error)
        assert "python" in msg
        assert "javascript" in msg

    def test_error_attributes(self):
        """Test error attributes."""
        error = UnsupportedLanguageError(".xyz", supported=["python"])
        assert error.identifier == ".xyz"
        assert error.supported == ["python"]


class TestClearFunctions:
    """Tests for clear_registry and clear_cache functions."""

    def test_clear_cache_removes_instances(self):
        """Test that clear_cache removes cached instances."""
        # Get an instance (will be cached)
        support1 = get_language_support(Language.PYTHON)

        # Clear cache
        clear_cache()

        # Get another instance (should be new)
        support2 = get_language_support(Language.PYTHON)

        assert support1 is not support2

    def test_clear_registry_removes_everything(self):
        """Test that clear_registry removes all registrations."""
        # Verify Python is registered
        assert is_language_supported(Language.PYTHON)

        # Clear registry
        clear_registry()

        # Now Python should not be supported
        assert not is_language_supported(Language.PYTHON)

        # Re-register all languages by importing
        from codeflash.languages.python.support import PythonSupport
        from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport

        # Need to manually register since decorator already ran
        register_language(PythonSupport)
        register_language(JavaScriptSupport)
        register_language(TypeScriptSupport)

        # Should be supported again
        assert is_language_supported(Language.PYTHON)
