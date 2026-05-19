from __future__ import annotations

import sys
from argparse import Namespace
from unittest.mock import Mock

from codeflash import tracer


def test_optimize_help_shows_tracer_help_for_javascript_projects(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "package.json").write_text('{"name": "js-app"}', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["codeflash", "--help"])

    tracer.main(Namespace(file=None))

    captured = capsys.readouterr()
    assert "--only-functions" in captured.out
    assert "Sub-commands" not in captured.out


def test_optimize_help_does_not_start_java_tracing(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    run_java_tracer = Mock()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["codeflash", "--help"])
    monkeypatch.setattr("codeflash.tracer._run_java_tracer", run_java_tracer)

    tracer.main(Namespace(file=None))

    captured = capsys.readouterr()
    run_java_tracer.assert_not_called()
    assert "--only-functions" in captured.out
    assert "No Java command provided" not in captured.out
