"""Tests for config_parser.py — monorepo language detection priority."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.code_utils.config_parser import parse_config_file


class TestMonorepoConfigPriority:
    """Verify that closer config files win over parent Java build files in monorepos."""

    def test_closer_package_json_wins_over_parent_pom_xml(self, tmp_path: Path) -> None:
        """In monorepo/frontend/, a local package.json should win over a parent pom.xml."""
        # Parent Java project
        (tmp_path / "pom.xml").write_text("<project></project>", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        # Child JS project
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "package.json").write_text(
            json.dumps({"name": "frontend", "codeflash": {"moduleRoot": "src"}}),
            encoding="utf-8",
        )
        (frontend / "src").mkdir()

        with patch("codeflash.code_utils.config_parser.Path") as mock_path_cls:
            mock_path_cls.cwd.return_value = frontend
            # find_package_json also uses Path.cwd; mock it at the source
            with patch("codeflash.code_utils.config_js.Path") as mock_js_path_cls:
                mock_js_path_cls.cwd.return_value = frontend
                # Also need to let normal Path operations work
                mock_path_cls.side_effect = Path
                mock_path_cls.cwd.return_value = frontend
                mock_js_path_cls.side_effect = Path
                mock_js_path_cls.cwd.return_value = frontend

                config, root = parse_config_file()

        # Should detect JS, not Java
        assert config.get("language") != "java", (
            "Closer package.json should take priority over parent pom.xml"
        )

    def test_java_wins_when_no_closer_js_config(self, tmp_path: Path) -> None:
        """When only a pom.xml exists (no package.json/pyproject.toml closer), Java config wins."""
        (tmp_path / "pom.xml").write_text("<project></project>", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)

        with patch("codeflash.code_utils.config_parser.Path") as mock_path_cls:
            mock_path_cls.side_effect = Path
            mock_path_cls.cwd.return_value = tmp_path
            with patch("codeflash.code_utils.config_js.Path") as mock_js_path_cls:
                mock_js_path_cls.side_effect = Path
                mock_js_path_cls.cwd.return_value = tmp_path

                config, root = parse_config_file()

        assert config.get("language") == "java"

    def test_same_level_package_json_wins_over_pom_xml(self, tmp_path: Path) -> None:
        """When pom.xml and package.json are at the same level, package.json wins (more specific)."""
        (tmp_path / "pom.xml").write_text("<project></project>", encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "mixed-project", "codeflash": {"moduleRoot": "src"}}),
            encoding="utf-8",
        )

        with patch("codeflash.code_utils.config_parser.Path") as mock_path_cls:
            mock_path_cls.side_effect = Path
            mock_path_cls.cwd.return_value = tmp_path
            with patch("codeflash.code_utils.config_js.Path") as mock_js_path_cls:
                mock_js_path_cls.side_effect = Path
                mock_js_path_cls.cwd.return_value = tmp_path

                config, root = parse_config_file()

        assert config.get("language") != "java", (
            "Same-level package.json should take priority over pom.xml"
        )
