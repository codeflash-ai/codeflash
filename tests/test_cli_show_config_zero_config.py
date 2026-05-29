from __future__ import annotations

from pathlib import Path

from codeflash.cli_cmds.cli import _handle_show_config
from codeflash.cli_cmds.console import console


def _capture_show_config(monkeypatch, project_root: Path) -> str:
    monkeypatch.chdir(project_root)
    with console.capture() as capture:
        _handle_show_config()
    return capture.get()


def test_show_config_reports_zero_config_java_projects(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.0</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
""",
        encoding="utf-8",
    )
    (tmp_path / "src" / "main" / "java").mkdir(parents=True)
    (tmp_path / "src" / "test" / "java").mkdir(parents=True)

    output = _capture_show_config(monkeypatch, tmp_path)

    assert "Codeflash Configuration" in output
    assert "Auto-detected (zero-config)" in output
    assert "Config source:" in output
    assert "Config file:" not in output
    assert "junit5" in output


def test_show_config_reports_zero_config_go_projects(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.21\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

    output = _capture_show_config(monkeypatch, tmp_path)

    assert "Codeflash Configuration" in output
    assert "Auto-detected (zero-config)" in output
    assert "Config source:" in output
    assert "Config file:" not in output
    assert "go-test" in output
