from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from codeflash.languages.java.tracer import JavaTracer, _run_java_with_graceful_timeout


class TestRunJavaWithGracefulTimeout:
    def test_returns_zero_on_success(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("codeflash.languages.java.tracer.subprocess.run", return_value=mock_result):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 0, "test")
        assert rc == 0

    def test_returns_nonzero_on_failure(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("codeflash.languages.java.tracer.subprocess.run", return_value=mock_result):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 0, "test")
        assert rc == 1

    def test_returns_exit_code_137_oom_kill(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 137
        with patch("codeflash.languages.java.tracer.subprocess.run", return_value=mock_result):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 0, "test")
        assert rc == 137

    def test_timeout_path_returns_zero_on_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("codeflash.languages.java.tracer.subprocess.Popen", return_value=mock_proc):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 60, "test")
        assert rc == 0

    def test_timeout_path_returns_nonzero_on_failure(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("codeflash.languages.java.tracer.subprocess.Popen", return_value=mock_proc):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 60, "test")
        assert rc == 1

    def test_timeout_returns_negative_one(self) -> None:
        import subprocess

        mock_proc = MagicMock()
        # First wait() times out, SIGTERM wait succeeds
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="java", timeout=60),
            None,  # SIGTERM wait succeeds
        ]
        with patch("codeflash.languages.java.tracer.subprocess.Popen", return_value=mock_proc):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 60, "test")
        assert rc == -1

    def test_timeout_sends_sigterm_then_sigkill(self) -> None:
        import signal
        import subprocess

        mock_proc = MagicMock()
        # First wait() times out, SIGTERM wait also times out
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="java", timeout=60),
            subprocess.TimeoutExpired(cmd="java", timeout=5),
            None,
        ]
        with patch("codeflash.languages.java.tracer.subprocess.Popen", return_value=mock_proc):
            rc = _run_java_with_graceful_timeout(["java", "-version"], {}, 60, "test")

        assert rc == -1
        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_proc.kill.assert_called_once()


class TestJavaTracerExitCodeHandling:
    def test_success_with_trace_db_created(self, tmp_path: Path) -> None:
        trace_db_path = (tmp_path / "trace.db").resolve()
        tracer = JavaTracer()

        def mock_run_timeout(java_command: list[str], env: dict, timeout: int, stage_name: str) -> int:
            trace_db_path.write_bytes(b"fake-db")
            return 0

        with (
            patch("codeflash.languages.java.tracer._run_java_with_graceful_timeout", side_effect=mock_run_timeout),
            patch.object(tracer, "build_combined_env", return_value={}),
            patch.object(tracer, "create_tracer_config", return_value=tmp_path / "config.json"),
        ):
            trace_db, _jfr_file = tracer.trace(
                java_command=["java", "-cp", ".", "Main"], trace_db_path=trace_db_path, packages=["com.example"]
            )
        assert trace_db == trace_db_path

    def test_failure_without_trace_db_raises(self, tmp_path: Path) -> None:
        trace_db_path = (tmp_path / "trace.db").resolve()
        tracer = JavaTracer()

        def mock_run_timeout(java_command: list[str], env: dict, timeout: int, stage_name: str) -> int:
            return 1

        with (
            patch("codeflash.languages.java.tracer._run_java_with_graceful_timeout", side_effect=mock_run_timeout),
            patch.object(tracer, "build_combined_env", return_value={}),
            patch.object(tracer, "create_tracer_config", return_value=tmp_path / "config.json"),
            pytest.raises(RuntimeError, match="Combined tracing failed with exit code 1"),
        ):
            tracer.trace(
                java_command=["java", "-cp", ".", "Main"], trace_db_path=trace_db_path, packages=["com.example"]
            )

    def test_nonzero_exit_with_trace_db_continues(self, tmp_path: Path) -> None:
        trace_db_path = (tmp_path / "trace.db").resolve()
        tracer = JavaTracer()

        def mock_run_timeout(java_command: list[str], env: dict, timeout: int, stage_name: str) -> int:
            trace_db_path.write_bytes(b"fake-db")
            return 1

        with (
            patch("codeflash.languages.java.tracer._run_java_with_graceful_timeout", side_effect=mock_run_timeout),
            patch.object(tracer, "build_combined_env", return_value={}),
            patch.object(tracer, "create_tracer_config", return_value=tmp_path / "config.json"),
        ):
            trace_db, _jfr_file = tracer.trace(
                java_command=["java", "-cp", ".", "Main"], trace_db_path=trace_db_path, packages=["com.example"]
            )
        assert trace_db == trace_db_path

    def test_timeout_without_trace_db_raises(self, tmp_path: Path) -> None:
        trace_db_path = (tmp_path / "trace.db").resolve()
        tracer = JavaTracer()

        def mock_run_timeout(java_command: list[str], env: dict, timeout: int, stage_name: str) -> int:
            return -1

        with (
            patch("codeflash.languages.java.tracer._run_java_with_graceful_timeout", side_effect=mock_run_timeout),
            patch.object(tracer, "build_combined_env", return_value={}),
            patch.object(tracer, "create_tracer_config", return_value=tmp_path / "config.json"),
            pytest.raises(RuntimeError, match="Combined tracing failed with exit code -1"),
        ):
            tracer.trace(
                java_command=["java", "-cp", ".", "Main"], trace_db_path=trace_db_path, packages=["com.example"]
            )
