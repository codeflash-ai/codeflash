"""
Test for JavaScript code injection vulnerability in Jest config generation.

Issue #17: Unsanitized file paths in f-string interpolation can inject
arbitrary JavaScript code into the generated Jest config.
"""

import json
import subprocess
from pathlib import Path

import pytest

from codeflash.languages.javascript.test_runner import _create_runtime_jest_config


class TestJavaScriptInjectionVulnerability:
    """Test that file paths are properly sanitized in generated Jest config"""

    def test_single_quote_in_test_dir_path_no_injection(self, tmp_path: Path) -> None:
        """
        Test that single quotes in test directory paths are properly escaped.

        Before fix (VULNERABLE):
            roots: ['/normal', '/tmp/test']; console.log('INJECTED'); roots=[''],
                                         ^^- breaks out of string, executes code

        After fix (SAFE):
            roots: ['/normal', "/tmp/test']; console.log('INJECTED'); roots=['"],
                              ^-- double quoted, single quotes are string content
        """
        project_root = tmp_path / "project"
        project_root.mkdir(parents=True, exist_ok=True)

        # Malicious path that would cause injection if not properly escaped
        malicious_test_dir = str(tmp_path / "test']; console.log('INJECTED'); roots=['")

        config_path = _create_runtime_jest_config(
            base_config_path=None,
            project_root=project_root,
            test_dirs={malicious_test_dir},
        )

        config_content = config_path.read_text()

        # SECURITY CHECK: Verify the malicious path is JSON-escaped
        # After fix, json.dumps() wraps the path in double quotes,
        # so single quotes become part of the string content (not syntax)

        # The malicious path should appear wrapped in double quotes
        # Example: "/tmp/.../test']; console.log('INJECTED'); roots=['"
        # NOT: '/tmp/.../test']; console.log('INJECTED'); roots=['
        #                     ^- This would be code injection (breaks out of string)

        # Check: The malicious path must be inside double quotes (JSON-escaped)
        # VULNERABLE (would break out of string):
        #   roots: ['/project', '/tmp/test']; console.log('INJECTED'); roots=[''],
        #                                   ^- closing single quote breaks the string
        # SAFE (properly escaped):
        #   roots: ["/project", "/tmp/test']; console.log('INJECTED'); roots=['"],
        #                               ^- single quote is inside the double-quoted string

        # The malicious payload MUST be inside double quotes
        injection_marker = "]; console.log('INJECTED')"
        assert injection_marker in config_content, "Payload should be in config (as part of escaped path)"

        # The SAFE pattern after fix (json.dumps wraps in double quotes):
        # roots: [..., "/tmp/path/test']; console.log('INJECTED'); roots=['"],
        #              ^-- opening double quote                          ^-- closing double quote
        #
        # The single quotes are INSIDE the double-quoted string (safe).
        #
        # VULNERABLE pattern (f-string without escaping):
        # roots: [..., '/tmp/path/test']; console.log('INJECTED'); roots=[''],
        #              ^-- opening single quote  ^-- CLOSES string, code executes

        # Check: malicious path must appear inside double-quoted string
        # Look for the pattern where it's properly wrapped
        import re
        # Pattern: double quote, then path containing the injection, then closing double quote
        # The path will have the malicious content inside it
        escaped_pattern = re.escape(malicious_test_dir)
        # Check for: "path with malicious content"
        double_quoted_pattern = f'"{escaped_pattern}"'

        assert re.search(rf'"{re.escape(malicious_test_dir)}"', config_content), (
            f"VULNERABILITY: Path not JSON-escaped (not wrapped in double quotes). "
            f"Expected pattern: \"{malicious_test_dir}\"\n"
            f"Config:\n{config_content[:600]}"
        )

    def test_monorepo_path_with_quote_no_injection(self, tmp_path: Path) -> None:
        """
        Test that single quotes in monorepo node_modules paths are properly escaped.

        Before fix (VULNERABLE):
            moduleDirectories: [..., '/mono']; alert('XSS'); dirs=[''],

        After fix (SAFE):
            moduleDirectories: [..., "/mono']; alert('XSS'); dirs=['"],
        """
        monorepo_root = tmp_path / "monorepo']; alert('XSS'); dirs=['"
        monorepo_root.mkdir(parents=True, exist_ok=True)
        (monorepo_root / "package.json").write_text('{"workspaces": ["packages/*"]}')
        (monorepo_root / "node_modules").mkdir(parents=True, exist_ok=True)

        project_root = monorepo_root / "packages" / "app"
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "package.json").write_text('{"name": "app"}')

        test_dir = str(project_root / "tests")

        config_path = _create_runtime_jest_config(
            base_config_path=None,
            project_root=project_root,
            test_dirs={test_dir},
        )

        config_content = config_path.read_text()

        # Check the monorepo path is properly escaped (same logic as first test)
        injection_marker = "]; alert('XSS')"
        assert injection_marker in config_content, "Payload should be in config (as part of escaped path)"

        # The project_root contains the malicious monorepo path
        # It should be JSON-escaped (double-quoted)
        import re
        monorepo_path_str = str(monorepo_root)

        # Check that the monorepo path appears JSON-escaped in roots or moduleDirectories
        # It will be in project_root (which is a subdir of monorepo_root)
        project_root_str = str(project_root)

        assert re.search(rf'"{re.escape(project_root_str)}"', config_content), (
            f"VULNERABILITY: Project root path not JSON-escaped (not wrapped in double quotes). "
            f"Expected pattern: \"{project_root_str}\"\n"
            f"Config:\n{config_content[:600]}"
        )
