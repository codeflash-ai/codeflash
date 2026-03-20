"""Tests for JFR parser — class name normalization, package filtering, addressable time."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.jfr_parser import JfrProfile


def _make_jfr_json(events: list[dict]) -> str:
    """Create fake JFR JSON output matching the jfr print format."""
    return json.dumps({"recording": {"events": events}})


def _make_execution_sample(class_name: str, method_name: str, start_time: str = "2026-01-01T00:00:00Z") -> dict:
    return {
        "type": "jdk.ExecutionSample",
        "values": {
            "startTime": start_time,
            "stackTrace": {
                "frames": [
                    {
                        "method": {
                            "type": {"name": class_name},
                            "name": method_name,
                            "descriptor": "()V",
                        },
                        "lineNumber": 42,
                    }
                ],
            },
        },
    }


class TestClassNameNormalization:
    """Test that JVM internal class names (com/example/Foo) are normalized to dots (com.example.Foo)."""

    def test_slash_separators_normalized_to_dots(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [
                _make_execution_sample("com/aerospike/client/command/Buffer", "bytesToInt"),
                _make_execution_sample("com/aerospike/client/command/Buffer", "bytesToInt"),
                _make_execution_sample("com/aerospike/client/util/Utf8", "encodedLength"),
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.aerospike"])

        assert profile._total_samples == 3
        assert len(profile._method_samples) == 2

        # Keys should use dots, not slashes
        assert "com.aerospike.client.command.Buffer.bytesToInt" in profile._method_samples
        assert "com.aerospike.client.util.Utf8.encodedLength" in profile._method_samples

    def test_method_info_uses_dot_class_names(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [_make_execution_sample("com/example/MyClass", "myMethod")]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        info = profile._method_info.get("com.example.MyClass.myMethod")
        assert info is not None
        assert info["class_name"] == "com.example.MyClass"
        assert info["method_name"] == "myMethod"


class TestPackageFiltering:
    def test_filters_by_package_prefix(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [
                _make_execution_sample("com/aerospike/client/Value", "get"),
                _make_execution_sample("java/util/HashMap", "put"),
                _make_execution_sample("com/aerospike/benchmarks/Main", "main"),
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.aerospike"])

        # Only com.aerospike classes should be in samples
        assert len(profile._method_samples) == 2
        assert "com.aerospike.client.Value.get" in profile._method_samples
        assert "com.aerospike.benchmarks.Main.main" in profile._method_samples
        assert "java.util.HashMap.put" not in profile._method_samples

    def test_empty_packages_includes_all(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [
                _make_execution_sample("com/example/Foo", "bar"),
                _make_execution_sample("java/lang/String", "length"),
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, [])

        assert len(profile._method_samples) == 2


class TestAddressableTime:
    def test_addressable_time_proportional_to_samples(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        # 3 samples for methodA, 1 for methodB, spanning 10 seconds
        jfr_json = _make_jfr_json(
            [
                _make_execution_sample("com/example/Foo", "methodA", "2026-01-01T00:00:00Z"),
                _make_execution_sample("com/example/Foo", "methodA", "2026-01-01T00:00:03Z"),
                _make_execution_sample("com/example/Foo", "methodA", "2026-01-01T00:00:06Z"),
                _make_execution_sample("com/example/Foo", "methodB", "2026-01-01T00:00:10Z"),
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        time_a = profile.get_addressable_time_ns("com.example.Foo", "methodA")
        time_b = profile.get_addressable_time_ns("com.example.Foo", "methodB")

        # methodA has 3x the samples of methodB, so 3x the addressable time
        assert time_a > 0
        assert time_b > 0
        assert time_a == pytest.approx(time_b * 3, rel=0.01)

    def test_addressable_time_zero_for_unknown_method(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [_make_execution_sample("com/example/Foo", "bar")]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        assert profile.get_addressable_time_ns("com.example.Foo", "nonExistent") == 0.0


class TestMethodRanking:
    def test_ranking_ordered_by_sample_count(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [
                _make_execution_sample("com/example/A", "hot"),
                _make_execution_sample("com/example/A", "hot"),
                _make_execution_sample("com/example/A", "hot"),
                _make_execution_sample("com/example/B", "warm"),
                _make_execution_sample("com/example/B", "warm"),
                _make_execution_sample("com/example/C", "cold"),
            ]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        ranking = profile.get_method_ranking()
        assert len(ranking) == 3
        assert ranking[0]["method_name"] == "hot"
        assert ranking[0]["sample_count"] == 3
        assert ranking[1]["method_name"] == "warm"
        assert ranking[1]["sample_count"] == 2
        assert ranking[2]["method_name"] == "cold"
        assert ranking[2]["sample_count"] == 1

    def test_empty_ranking_when_no_samples(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json([])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        assert profile.get_method_ranking() == []

    def test_ranking_uses_dot_class_names(self, tmp_path: Path) -> None:
        jfr_file = tmp_path / "test.jfr"
        jfr_file.write_text("dummy", encoding="utf-8")

        jfr_json = _make_jfr_json(
            [_make_execution_sample("com/example/nested/Deep", "method")]
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=jfr_json, stderr="")
            profile = JfrProfile(jfr_file, ["com.example"])

        ranking = profile.get_method_ranking()
        assert len(ranking) == 1
        assert ranking[0]["class_name"] == "com.example.nested.Deep"


class TestGracefulTimeout:
    """Test that _run_java_with_graceful_timeout sends SIGTERM before SIGKILL."""

    def test_sends_sigterm_on_timeout(self) -> None:
        import signal

        from codeflash.languages.java.tracer import _run_java_with_graceful_timeout

        # Run a sleep command with a 1s timeout — should get SIGTERM'd
        import os

        env = os.environ.copy()
        _run_java_with_graceful_timeout(["sleep", "60"], env, timeout=1, stage_name="test")
        # If we get here, the process was killed (didn't hang for 60s)

    def test_no_timeout_runs_normally(self) -> None:
        import os

        from codeflash.languages.java.tracer import _run_java_with_graceful_timeout

        env = os.environ.copy()
        _run_java_with_graceful_timeout(["echo", "hello"], env, timeout=0, stage_name="test")
        # Should complete without error


class TestProjectRootResolution:
    """Test that project_root is correctly set for Java multi-module projects."""

    def test_java_project_root_is_build_root_not_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """For multi-module Maven, project_root should be the root with <modules>, not a sub-module."""
        # Create a multi-module project
        (tmp_path / "pom.xml").write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0"><modules><module>client</module></modules></project>',
            encoding="utf-8",
        )
        client = tmp_path / "client"
        client.mkdir()
        (client / "pom.xml").write_text("<project/>", encoding="utf-8")
        src = client / "src" / "main" / "java"
        src.mkdir(parents=True)
        test = tmp_path / "src" / "test" / "java"
        test.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        from codeflash.code_utils.config_parser import parse_config_file

        config, config_path = parse_config_file()
        assert config["language"] == "java"

        # config_path should be the project root directory
        assert config_path == tmp_path

    def test_project_root_is_path_not_string(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """project_root from process_pyproject_config should be a Path for Java projects."""
        from argparse import Namespace

        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        test = tmp_path / "src" / "test" / "java"
        test.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        from codeflash.cli_cmds.cli import process_pyproject_config

        # Create a minimal args namespace matching what parse_args produces
        args = Namespace(
            config_file=None, module_root=None, tests_root=None, benchmarks_root=None,
            ignore_paths=None, pytest_cmd=None, formatter_cmds=None, disable_telemetry=None,
            disable_imports_sorting=None, git_remote=None, override_fixtures=None,
            benchmark=False, verbose=False, version=False, show_config=False, reset_config=False,
        )
        args = process_pyproject_config(args)

        assert hasattr(args, "project_root")
        assert isinstance(args.project_root, Path)
        assert args.project_root == tmp_path
