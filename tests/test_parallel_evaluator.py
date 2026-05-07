"""Integration tests for the parallel candidate evaluation infrastructure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import pytest

from codeflash.either import Failure, Success, is_successful
from codeflash.languages.function_optimizer import CandidateNode
from codeflash.optimization.parallel_evaluator import EvalFailure, ParallelCandidateEvaluator


class TestWorktreePoolLifecycle:
    def test_creates_n_worktrees_and_cleans_up(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from codeflash.code_utils.worktree_pool import WorktreePool

        pool_size = 3
        base_dir = tmp_path.resolve() / "worktrees"

        # The pool needs a git root. We use the codeflash repo itself.
        repo_root = Path(__file__).resolve().parents[1]

        async def _run() -> None:
            with patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root):
                pool = WorktreePool(pool_size=pool_size, base_dir=base_dir)
                async with pool:
                    assert len(pool._slots) == pool_size
                    for slot in pool._slots:
                        assert slot.path.exists()
                        assert slot.path.is_dir()

                # After cleanup, slots are cleared
                assert len(pool._slots) == 0

        anyio.run(_run)

    def test_partial_pool_initialization(self, tmp_path: Path) -> None:
        """Pool operates at reduced capacity if some slots fail to create."""
        from unittest.mock import patch

        from codeflash.code_utils.worktree_pool import WorktreePool

        pool_size = 3
        base_dir = tmp_path.resolve() / "worktrees"
        repo_root = Path(__file__).resolve().parents[1]

        call_count = 0

        original_create_slot = WorktreePool._create_slot

        async def failing_create_slot(self: Any, index: int) -> Any:
            nonlocal call_count
            call_count += 1
            if index == 1:
                raise RuntimeError("Simulated git worktree failure")
            return await original_create_slot(self, index)

        async def _run() -> None:
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(WorktreePool, "_create_slot", failing_create_slot),
            ):
                async with WorktreePool(pool_size=pool_size, base_dir=base_dir) as pool:
                    assert len(pool._slots) == 2
                    slot = await pool.acquire()
                    assert slot.index != 1
                    await pool.release(slot)

        anyio.run(_run)

    def test_acquire_release_round_trip(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from codeflash.code_utils.worktree_pool import WorktreePool

        pool_size = 2
        base_dir = tmp_path.resolve() / "worktrees"
        repo_root = Path(__file__).resolve().parents[1]

        async def _run() -> None:
            with patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root):
                async with WorktreePool(pool_size=pool_size, base_dir=base_dir) as pool:
                    slot1 = await pool.acquire()
                    slot2 = await pool.acquire()

                    # Both slots should be distinct
                    assert slot1.index != slot2.index
                    assert slot1.path != slot2.path

                    # Release one and re-acquire it
                    await pool.release(slot1)
                    reacquired = await pool.acquire()
                    assert reacquired.index == slot1.index

                    await pool.release(slot2)
                    await pool.release(reacquired)

        anyio.run(_run)


class TestWorktreeSlotFileIsolation:
    def test_write_to_one_slot_does_not_affect_another(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from codeflash.code_utils.worktree_pool import WorktreePool

        pool_size = 2
        base_dir = tmp_path.resolve() / "worktrees"
        repo_root = Path(__file__).resolve().parents[1]
        test_file = repo_root / "codeflash" / "__init__.py"

        async def _run() -> None:
            with patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root):
                async with WorktreePool(pool_size=pool_size, base_dir=base_dir) as pool:
                    slot_a = await pool.acquire()
                    slot_b = await pool.acquire()

                    sentinel = "# SLOT_A_SENTINEL_CONTENT\n"
                    await slot_a.write_candidate(test_file, sentinel)

                    # slot_b's mirror of the same file should NOT contain the sentinel
                    mirrored_b = slot_b.mirror(test_file)
                    content_b = mirrored_b.read_text(encoding="utf-8")
                    assert sentinel not in content_b

                    # slot_a's mirror should contain it
                    mirrored_a = slot_a.mirror(test_file)
                    content_a = mirrored_a.read_text(encoding="utf-8")
                    assert content_a == sentinel

                    # Main tree should be unaffected
                    main_content = test_file.read_text(encoding="utf-8")
                    assert sentinel not in main_content

                    await pool.release(slot_a)
                    await pool.release(slot_b)

        anyio.run(_run)


class TestAsyncExecuteTestSubprocess:
    def test_runs_simple_command(self) -> None:
        from codeflash.languages.python.test_runner import async_execute_test_subprocess

        cwd = Path(__file__).resolve().parent

        async def _run() -> subprocess.CompletedProcess[str]:
            return await async_execute_test_subprocess(
                cmd_list=[sys.executable, "-c", "print('hello world')"], cwd=cwd, env=None, timeout=30
            )

        result = anyio.run(_run)
        assert result.returncode == 0
        assert "hello world" in result.stdout

    def test_captures_stderr(self) -> None:
        from codeflash.languages.python.test_runner import async_execute_test_subprocess

        cwd = Path(__file__).resolve().parent

        async def _run() -> subprocess.CompletedProcess[str]:
            return await async_execute_test_subprocess(
                cmd_list=[sys.executable, "-c", "import sys; sys.stderr.write('err_msg\\n')"],
                cwd=cwd,
                env=None,
                timeout=30,
            )

        result = anyio.run(_run)
        assert "err_msg" in result.stderr

    def test_timeout_raises(self) -> None:
        from codeflash.languages.python.test_runner import async_execute_test_subprocess

        cwd = Path(__file__).resolve().parent

        async def _run() -> subprocess.CompletedProcess[str]:
            return await async_execute_test_subprocess(
                cmd_list=[sys.executable, "-c", "import time; time.sleep(60)"], cwd=cwd, env=None, timeout=1
            )

        with pytest.raises(subprocess.TimeoutExpired):
            anyio.run(_run)


class TestParallelCandidateEvaluator:
    """Unit tests for the evaluator with mocked worktree operations."""

    def _make_candidate_node(self, opt_id: str = "cand_1") -> CandidateNode:
        from codeflash.models.models import CodeString, CodeStringsMarkdown, OptimizedCandidate
        from codeflash.models.shared_types import OptimizedCandidateSource

        source_code = CodeStringsMarkdown(code_strings=[CodeString(code="def f(): pass", file_path=Path("test.py"))])
        candidate = OptimizedCandidate(
            source_code=source_code,
            explanation="test optimization",
            optimization_id=opt_id,
            source=OptimizedCandidateSource.OPTIMIZE,
        )
        return CandidateNode(candidate)

    def _make_optimizer_mock(self, tmp_path: Path) -> MagicMock:
        opt = MagicMock()
        opt.function_to_optimize.file_path = str(tmp_path / "src" / "module.py")
        opt.function_to_optimize_source_code = "def f(): pass"
        opt.test_files.test_files = []
        opt.args.project_root = str(tmp_path)
        opt.test_cfg = MagicMock()
        opt.get_test_env.return_value = {"PATH": "/usr/bin"}
        opt.language_support.build_pytest_cmd.return_value = [sys.executable, "-m", "pytest"]
        opt.replace_function_and_helpers_with_optimized_code.return_value = True
        opt.write_code_and_helpers = MagicMock()
        return opt

    def test_code_replacement_failure_returns_eval_failure(self, tmp_path: Path) -> None:
        opt = self._make_optimizer_mock(tmp_path)
        opt.replace_function_and_helpers_with_optimized_code.return_value = False

        node = self._make_candidate_node()
        evaluator = ParallelCandidateEvaluator(opt, pool_size=1)

        repo_root = Path(__file__).resolve().parents[1]

        async def _run() -> list:  # type: ignore[type-arg]
            with patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root):
                return await evaluator.evaluate_candidates(
                    candidates=[(node, 0, None)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        results = anyio.run(_run)
        assert len(results) == 1
        _, result = results[0]
        assert result is not None
        assert not is_successful(result)
        failure = result.failure()
        assert isinstance(failure, EvalFailure)
        assert "Code replacement failed" in failure.message
        assert failure.diffs == []

    def test_behavioral_mismatch_carries_diffs(self, tmp_path: Path) -> None:
        from codeflash.models.models import TestDiff, TestDiffScope

        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "module.py").write_text("def f(): pass", encoding="utf-8")

        node = self._make_candidate_node()
        evaluator = ParallelCandidateEvaluator(opt, pool_size=1)

        repo_root = Path(__file__).resolve().parents[1]
        mock_diffs = [TestDiff(scope=TestDiffScope.DID_PASS, original_pass=True, candidate_pass=False)]

        async def _run() -> list:  # type: ignore[type-arg]
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(
                    ParallelCandidateEvaluator,
                    "_run_behavioral",
                    return_value=Failure(EvalFailure(message="Behavioral mismatch: 1 diffs", diffs=mock_diffs)),  # type: ignore[arg-type]
                ),
            ):
                return await evaluator.evaluate_candidates(
                    candidates=[(node, 0, None)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        results = anyio.run(_run)
        _, result = results[0]
        assert not is_successful(result)
        failure = result.failure()
        assert len(failure.diffs) == 1
        assert failure.diffs[0].scope == TestDiffScope.DID_PASS

    def test_successful_candidate_returns_result(self, tmp_path: Path) -> None:
        from codeflash.optimization.parallel_evaluator import _BehavioralPass

        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "module.py").write_text("def f(): pass", encoding="utf-8")

        node = self._make_candidate_node()
        evaluator = ParallelCandidateEvaluator(opt, pool_size=1)

        repo_root = Path(__file__).resolve().parents[1]
        mock_result = MagicMock()
        mock_result.best_test_runtime = 5000

        mock_behavior_results = MagicMock()

        async def mock_behavioral(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            return Success(
                _BehavioralPass(
                    candidate_index=0,
                    perf_test_files=[],
                    test_env={},
                    pytest_cmd_list=[],
                    behavior_test_results=mock_behavior_results,
                    fto_code="def f(): pass",
                    helper_codes={},
                    fto_file_path=Path("/tmp/module.py"),
                )
            )

        async def _run() -> list:  # type: ignore[type-arg]
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(ParallelCandidateEvaluator, "_run_behavioral", mock_behavioral),
                patch.object(ParallelCandidateEvaluator, "_benchmark_phase", return_value=Success(mock_result)),
            ):
                return await evaluator.evaluate_candidates(
                    candidates=[(node, 0, None)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        results = anyio.run(_run)
        _, result = results[0]
        assert is_successful(result)
        assert result.unwrap().best_test_runtime == 5000

    def test_multiple_candidates_evaluated_concurrently(self, tmp_path: Path) -> None:
        from codeflash.optimization.parallel_evaluator import _BehavioralPass

        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "module.py").write_text("def f(): pass", encoding="utf-8")

        nodes = [self._make_candidate_node(f"cand_{i}") for i in range(3)]
        evaluator = ParallelCandidateEvaluator(opt, pool_size=3)

        repo_root = Path(__file__).resolve().parents[1]
        mock_result = MagicMock()
        mock_result.best_test_runtime = 1000

        behavioral_call_count = 0
        mock_behavior_results = MagicMock()

        async def mock_behavioral(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            nonlocal behavioral_call_count
            behavioral_call_count += 1
            return Success(
                _BehavioralPass(
                    candidate_index=0,
                    perf_test_files=[],
                    test_env={},
                    pytest_cmd_list=[],
                    behavior_test_results=mock_behavior_results,
                    fto_code="def f(): pass",
                    helper_codes={},
                    fto_file_path=Path("/tmp/module.py"),
                )
            )

        benchmark_call_count = 0

        async def mock_benchmark(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            nonlocal benchmark_call_count
            benchmark_call_count += 1
            return Success(mock_result)

        async def _run() -> list:  # type: ignore[type-arg]
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(ParallelCandidateEvaluator, "_run_behavioral", mock_behavioral),
                patch.object(ParallelCandidateEvaluator, "_benchmark_phase", mock_benchmark),
            ):
                return await evaluator.evaluate_candidates(
                    candidates=[(n, i, None) for i, n in enumerate(nodes)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        results = anyio.run(_run)
        assert len(results) == 3
        assert behavioral_call_count == 3
        assert benchmark_call_count == 3
        for _, result in results:
            assert is_successful(result)

    def test_benchmark_phase_restages_candidate_code(self, tmp_path: Path) -> None:
        """Phase 2 must write fto_code and helper_codes to the slot before running benchmarks."""
        from codeflash.optimization.parallel_evaluator import _BehavioralPass

        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "module.py").write_text("def f(): pass", encoding="utf-8")

        node = self._make_candidate_node()
        evaluator = ParallelCandidateEvaluator(opt, pool_size=1)

        repo_root = Path(__file__).resolve().parents[1]
        fto_code = "def f(): return 42  # optimized"
        helper_path = tmp_path / "src" / "helpers.py"
        helper_codes = {helper_path: "HELPER_CODE = True"}

        write_calls: list[tuple[Path, str]] = []

        async def tracking_write_candidate(self_slot: object, file_path: Path, code: str) -> None:
            write_calls.append((file_path, code))

        async def mock_behavioral(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            return Success(
                _BehavioralPass(
                    candidate_index=0,
                    perf_test_files=[],
                    test_env={"PATH": "/usr/bin"},
                    pytest_cmd_list=[sys.executable, "-m", "pytest"],
                    behavior_test_results=MagicMock(),
                    fto_code=fto_code,
                    helper_codes=helper_codes,
                    fto_file_path=Path(opt.function_to_optimize.file_path),
                )
            )

        async def _run() -> list:  # type: ignore[type-arg]
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(ParallelCandidateEvaluator, "_run_behavioral", mock_behavioral),
                patch(
                    "codeflash.code_utils.worktree_pool.WorktreeSlot.write_candidate", tracking_write_candidate
                ),
                patch(
                    "codeflash.languages.python.test_runner.async_execute_test_subprocess",
                    return_value=MagicMock(returncode=0, stdout="", stderr=""),
                ),
                patch(
                    "codeflash.verification.parse_test_output.parse_test_xml",
                    return_value=MagicMock(test_results=[MagicMock()], effective_loop_count=lambda: 10, total_passed_runtime=lambda: 5000),
                ),
            ):
                return await evaluator.evaluate_candidates(
                    candidates=[(node, 0, None)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        anyio.run(_run)

        written_codes = {p: c for p, c in write_calls}
        assert Path(opt.function_to_optimize.file_path) in written_codes
        assert written_codes[Path(opt.function_to_optimize.file_path)] == fto_code
        assert helper_path in written_codes
        assert written_codes[helper_path] == "HELPER_CODE = True"

    def test_empty_candidates_returns_empty(self, tmp_path: Path) -> None:
        opt = self._make_optimizer_mock(tmp_path)
        evaluator = ParallelCandidateEvaluator(opt, pool_size=1)
        repo_root = Path(__file__).resolve().parents[1]

        async def _run() -> list:  # type: ignore[type-arg]
            with patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root):
                return await evaluator.evaluate_candidates(
                    candidates=[],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        results = anyio.run(_run)
        assert results == []

    def test_replace_and_capture_restores_on_failure(self, tmp_path: Path) -> None:
        """_replace_and_capture must restore original code even when replacement raises."""
        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        original_code = "def f(): pass"
        (tmp_path / "src" / "module.py").write_text(original_code, encoding="utf-8")

        opt.replace_function_and_helpers_with_optimized_code.side_effect = ValueError("bad code")

        result = ParallelCandidateEvaluator._replace_and_capture(
            opt, MagicMock(), MagicMock(), {}
        )
        assert result is None
        opt.write_code_and_helpers.assert_called_once_with(
            opt.function_to_optimize_source_code, {}, opt.function_to_optimize.file_path
        )

    def test_more_candidates_than_slots_no_deadlock(self, tmp_path: Path) -> None:
        """Regression test: more passing candidates than pool slots must not deadlock."""
        from codeflash.optimization.parallel_evaluator import _BehavioralPass

        opt = self._make_optimizer_mock(tmp_path)
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "module.py").write_text("def f(): pass", encoding="utf-8")

        nodes = [self._make_candidate_node(f"cand_{i}") for i in range(6)]
        evaluator = ParallelCandidateEvaluator(opt, pool_size=2)

        repo_root = Path(__file__).resolve().parents[1]
        mock_result = MagicMock()
        mock_result.best_test_runtime = 2000
        mock_behavior_results = MagicMock()

        async def mock_behavioral(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            return Success(
                _BehavioralPass(
                    candidate_index=0,
                    perf_test_files=[],
                    test_env={},
                    pytest_cmd_list=[],
                    behavior_test_results=mock_behavior_results,
                    fto_code="def f(): pass",
                    helper_codes={},
                    fto_file_path=Path("/tmp/module.py"),
                )
            )

        async def mock_benchmark(self_eval: object, *args: object, **kwargs: object) -> Success:  # type: ignore[type-arg]
            return Success(mock_result)

        async def _run() -> list:  # type: ignore[type-arg]
            with (
                patch("codeflash.code_utils.worktree_pool.git_root_dir", return_value=repo_root),
                patch.object(ParallelCandidateEvaluator, "_run_behavioral", mock_behavioral),
                patch.object(ParallelCandidateEvaluator, "_benchmark_phase", mock_benchmark),
            ):
                return await evaluator.evaluate_candidates(
                    candidates=[(n, i, None) for i, n in enumerate(nodes)],
                    code_context=MagicMock(),
                    original_code_baseline=MagicMock(),
                    original_helper_code={},
                    file_path_to_helper_classes={},
                )

        # If this deadlocks, the test will timeout
        results = anyio.run(_run)
        assert len(results) == 6
        for _, result in results:
            assert is_successful(result)
