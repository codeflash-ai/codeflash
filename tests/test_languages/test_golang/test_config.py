from __future__ import annotations

from pathlib import Path

from codeflash.code_utils.config_parser import parse_config_file
from codeflash.languages.golang.config import detect_go_project, is_go_project

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "go_project"


class TestDetectGoProject:
    def test_detects_project(self) -> None:
        config = detect_go_project(FIXTURES_DIR)
        assert config is not None
        assert config.module_path == "github.com/example/myproject"
        assert config.go_version == "1.22.0"

    def test_no_go_mod(self, tmp_path: Path) -> None:
        config = detect_go_project(tmp_path)
        assert config is None

    def test_minimal_go_mod(self, tmp_path: Path) -> None:
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/minimal\n\ngo 1.21\n", encoding="utf-8")
        config = detect_go_project(tmp_path)
        assert config is not None
        assert config.module_path == "example.com/minimal"
        assert config.go_version == "1.21"

    def test_vendor_detection(self, tmp_path: Path) -> None:
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/vendored\n\ngo 1.22\n", encoding="utf-8")
        (tmp_path / "vendor").mkdir()
        config = detect_go_project(tmp_path)
        assert config is not None
        assert config.has_vendor is True


class TestIsGoProject:
    def test_with_go_mod(self) -> None:
        assert is_go_project(FIXTURES_DIR) is True

    def test_without_go_files(self, tmp_path: Path) -> None:
        assert is_go_project(tmp_path) is False

    def test_with_go_files_no_mod(self, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
        assert is_go_project(tmp_path) is True


class TestParseGoConfig:
    def test_parse_config_file_uses_go_test_metadata(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "go.mod").write_text("module example.com/minimal\n\ngo 1.21\n", encoding="utf-8")
        (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        config, config_path = parse_config_file()

        assert config_path == tmp_path
        assert config["language"] == "go"
        assert config["module_root"] == str(tmp_path.resolve())
        assert config["tests_root"] == str(tmp_path.resolve())
        assert config["pytest_cmd"] == "go test ./..."
        assert config["test_framework"] == "go-test"
