"""Tests for JavaScript requirements verification.

Tests the verify_requirements function that checks Node.js, npm, and test framework availability.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport


class TestVerifyRequirements:
    """Tests for JavaScriptSupport.verify_requirements()."""

    @pytest.fixture
    def js_support(self):
        """Create a JavaScriptSupport instance."""
        return JavaScriptSupport()

    @pytest.fixture
    def project_with_jest(self, tmp_path):
        """Create a project directory with Jest installed."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "jest").mkdir()
        (node_modules / "codeflash").mkdir()

        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"jest": "^29.0.0"},
                }
            )
        )
        return tmp_path

    @pytest.fixture
    def project_with_vitest(self, tmp_path):
        """Create a project directory with Vitest installed."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "vitest").mkdir()
        (node_modules / "codeflash").mkdir()

        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "devDependencies": {"vitest": "^2.0.0"},
                }
            )
        )
        return tmp_path

    @pytest.fixture
    def project_without_node_modules(self, tmp_path):
        """Create a project directory without node_modules."""
        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        return tmp_path

    @pytest.fixture
    def project_without_jest(self, tmp_path):
        """Create a project directory with node_modules but without Jest."""
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "some-other-package").mkdir()

        package_json = tmp_path / "package.json"
        package_json.write_text(json.dumps({"name": "test-project"}))
        return tmp_path

    def test_verify_requirements_success_with_jest(self, js_support, project_with_jest):
        """Test successful verification when Jest is installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_with_jest, "jest")

            assert success is True
            assert errors == []

    def test_verify_requirements_success_with_vitest(self, js_support, project_with_vitest):
        """Test successful verification when Vitest is installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_with_vitest, "vitest")

            assert success is True
            assert errors == []

    def test_verify_requirements_fails_without_node(self, js_support, project_with_jest):
        """Test verification fails when Node.js is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("node not found")

            success, errors = js_support.verify_requirements(project_with_jest, "jest")

            assert success is False
            assert len(errors) >= 1
            node_error_found = any("Node.js" in error for error in errors)
            assert node_error_found is True

    def test_verify_requirements_fails_without_npm(self, js_support, project_with_jest):
        """Test verification fails when npm is not available."""

        def mock_run_side_effect(cmd, **kwargs):
            if cmd[0] == "node":
                return MagicMock(returncode=0)
            if cmd[0] == "npm":
                raise FileNotFoundError("npm not found")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            success, errors = js_support.verify_requirements(project_with_jest, "jest")

            assert success is False
            npm_error_found = any("npm" in error for error in errors)
            assert npm_error_found is True

    def test_verify_requirements_fails_without_node_modules(self, js_support, project_without_node_modules):
        """Test verification fails when node_modules doesn't exist."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_without_node_modules, "jest")

            assert success is False
            assert len(errors) == 1
            expected_error = (
                f"node_modules not found in {project_without_node_modules}. "
                f"Please run 'npm install' to install dependencies."
            )
            assert errors[0] == expected_error

    def test_verify_requirements_fails_without_test_framework(self, js_support, project_without_jest):
        """Test verification fails when test framework is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_without_jest, "jest")

            assert success is False
            assert len(errors) == 1
            expected_error = "jest is not installed. Please run 'npm install --save-dev jest' to install it."
            assert errors[0] == expected_error

    def test_verify_requirements_returns_multiple_errors(self, js_support, project_without_node_modules):
        """Test that multiple errors can be returned."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("command not found")

            success, errors = js_support.verify_requirements(project_without_node_modules, "jest")

            assert success is False
            assert len(errors) >= 2
            # Should have errors for Node.js, npm, and node_modules
            error_text = " ".join(errors)
            assert "Node.js" in error_text
            assert "npm" in error_text

    def test_verify_requirements_vitest_not_installed(self, js_support, project_with_jest):
        """Test verification fails when Vitest is requested but only Jest is installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_with_jest, "vitest")

            assert success is False
            assert len(errors) == 1
            expected_error = "vitest is not installed. Please run 'npm install --save-dev vitest' to install it."
            assert errors[0] == expected_error

    def test_verify_requirements_jest_not_installed(self, js_support, project_with_vitest):
        """Test verification fails when Jest is requested but only Vitest is installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_with_vitest, "jest")

            assert success is False
            assert len(errors) == 1
            expected_error = "jest is not installed. Please run 'npm install --save-dev jest' to install it."
            assert errors[0] == expected_error


class TestVerifyRequirementsIntegration:
    """Integration tests for verify_requirements with real filesystem."""

    @pytest.fixture
    def js_support(self):
        """Create a JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_verify_on_real_vitest_project(self, js_support):
        """Test verification on the real vitest sample project."""
        project_root = Path(__file__).parent.parent.parent / "code_to_optimize" / "js" / "code_to_optimize_vitest"

        if not project_root.exists():
            pytest.skip("code_to_optimize_vitest directory not found")

        node_modules = project_root / "node_modules"
        if not node_modules.exists():
            pytest.skip("node_modules not installed in vitest project")

        # This test verifies the real project structure
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_root, "vitest")

            # If vitest is installed, should succeed
            vitest_installed = (node_modules / "vitest").exists()
            if vitest_installed:
                assert success is True
                assert errors == []
            else:
                assert success is False
                assert len(errors) >= 1

    def test_verify_on_real_jest_project(self, js_support):
        """Test verification on the real Jest sample project."""
        project_root = Path(__file__).parent.parent.parent / "code_to_optimize" / "js" / "code_to_optimize_ts"

        if not project_root.exists():
            pytest.skip("code_to_optimize_ts directory not found")

        node_modules = project_root / "node_modules"
        if not node_modules.exists():
            pytest.skip("node_modules not installed in jest project")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            success, errors = js_support.verify_requirements(project_root, "jest")

            jest_installed = (node_modules / "jest").exists()
            if jest_installed:
                assert success is True
                assert errors == []
            else:
                assert success is False
                assert len(errors) >= 1