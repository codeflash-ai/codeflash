# mypy: ignore-errors
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import INDIVIDUAL_TESTCASE_TIMEOUT, TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.code_utils.worktree_pool import WorktreePool, WorktreeSlot  # noqa: TC001
from codeflash.either import Failure, Success

if TYPE_CHECKING:
    from codeflash.either import Result
    from codeflash.languages.function_optimizer import CandidateNode, FunctionOptimizer
    from codeflash.models.models import (
        CandidateEvaluationContext,
        CodeOptimizationContext,
        OptimizedCandidate,
        OptimizedCandidateResult,
        OriginalCodeBaseline,
        TestDiff,
        TestResults,
    )


@dataclasses.dataclass(slots=True)
class EvalFailure:
    """Structured failure from parallel evaluation, carrying test diffs for repair."""

    message: str
    diffs: list[TestDiff] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class _BehavioralPass:
    """Intermediate result: candidate passed behavioral tests, ready for benchmarking."""

    candidate_index: int
    perf_test_files: list[str]
    test_env: dict[str, str]
    pytest_cmd_list: list[str]
    behavior_test_results: TestResults


class ParallelCandidateEvaluator:
    """Evaluates optimization candidates in parallel using git worktrees.

    Two-phase evaluation:
      Phase 1 (concurrent): behavioral correctness tests — slots released after each test
      Phase 2 (sequential): benchmarking — one candidate at a time for accurate timing
    """

    def __init__(self, optimizer: FunctionOptimizer, pool_size: int = 4) -> None:
        self._optimizer = optimizer
        self._pool_size = pool_size
        self._pool: WorktreePool | None = None

    async def evaluate_candidates(
        self,
        candidates: list[tuple[CandidateNode, int, str | None]],
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> list[tuple[CandidateNode, Result[OptimizedCandidateResult, EvalFailure] | None]]:
        """Evaluate candidates: behavioral tests concurrently, benchmarks sequentially."""
        results: list[tuple[CandidateNode, Result[OptimizedCandidateResult, EvalFailure] | None]] = [
            (node, None) for node, _, _ in candidates
        ]

        if not candidates:
            return results

        async with WorktreePool(pool_size=self._pool_size) as pool:
            self._pool = pool

            # Phase 1: concurrent behavioral tests (slots released after each test)
            behavioral_passes: list[tuple[int, CandidateNode, _BehavioralPass]] = []

            async with anyio.create_task_group() as tg:
                for i, (node, idx, _cached) in enumerate(candidates):
                    tg.start_soon(
                        self._behavioral_phase,
                        i,
                        node,
                        idx,
                        code_context,
                        original_code_baseline,
                        original_helper_code,
                        file_path_to_helper_classes,
                        results,
                        behavioral_passes,
                    )

            # Phase 2: sequential benchmarking (no CPU contention)
            for result_index, candidate_node, bp in behavioral_passes:
                slot = await pool.acquire()
                try:
                    bench_result = await self._benchmark_phase(slot, bp, original_code_baseline)
                    results[result_index] = (candidate_node, bench_result)
                except Exception as exc:
                    logger.error(f"Benchmark for {candidate_node.candidate.optimization_id} raised: {exc}")
                    results[result_index] = (candidate_node, Failure(EvalFailure(message=str(exc))))
                finally:
                    await pool.release(slot)

        return results

    async def _behavioral_phase(
        self,
        result_index: int,
        candidate_node: CandidateNode,
        candidate_index: int,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        results: list[tuple[CandidateNode, Result[OptimizedCandidateResult, EvalFailure] | None]],
        behavioral_passes: list[tuple[int, CandidateNode, _BehavioralPass]],
    ) -> None:
        """Run behavioral tests for a candidate. Slot is always released after the test."""
        assert self._pool is not None
        slot = await self._pool.acquire()
        try:
            outcome = await self._run_behavioral(
                slot=slot,
                candidate=candidate_node.candidate,
                candidate_index=candidate_index,
                code_context=code_context,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
            )
        except BaseException as exc:
            if not isinstance(exc, Exception):
                await self._pool.release(slot)
                raise
            logger.error(f"Candidate {candidate_node.candidate.optimization_id} raised: {exc}")
            results[result_index] = (candidate_node, Failure(EvalFailure(message=str(exc))))
            await self._pool.release(slot)
            return

        # Always release slot — Phase 2 re-acquires for benchmarking
        await self._pool.release(slot)

        if isinstance(outcome, Failure):
            results[result_index] = (candidate_node, outcome)
            return

        behavioral_passes.append((result_index, candidate_node, outcome.unwrap()))

    async def _run_behavioral(
        self,
        slot: WorktreeSlot,
        candidate: OptimizedCandidate,
        candidate_index: int,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
    ) -> Result[_BehavioralPass, EvalFailure]:
        """Run behavioral tests in a worktree. Returns pass info or failure."""
        opt = self._optimizer
        fto = opt.function_to_optimize

        candidate_files = await anyio.to_thread.run_sync(
            self._replace_and_capture, opt, code_context, candidate, original_helper_code
        )

        if candidate_files is None:
            return Failure(EvalFailure(message="Code replacement failed"))

        fto_code, helper_codes = candidate_files
        await slot.write_candidate(Path(fto.file_path), fto_code)
        for module_abspath, helper_code in helper_codes.items():
            await slot.write_candidate(module_abspath, helper_code)

        # Copy instrumented test files into the worktree
        behavior_test_files: list[str] = []
        perf_test_files: list[str] = []
        for file in opt.test_files.test_files:
            src = file.instrumented_behavior_file_path
            if src.exists():
                await slot.write_candidate(src, src.read_text(encoding="utf-8"))
            behavior_test_files.append(str(slot.mirror(src)))

            if file.benchmarking_file_path and file.benchmarking_file_path.exists():
                await slot.write_candidate(
                    file.benchmarking_file_path, file.benchmarking_file_path.read_text(encoding="utf-8")
                )
                perf_test_files.append(str(slot.mirror(file.benchmarking_file_path)))

        # Build test environment and command
        test_env = opt.get_test_env(
            codeflash_loop_index=0, codeflash_test_iteration=candidate_index, codeflash_tracer_disable=1
        )
        worktree_project_root = slot.mirror(Path(opt.args.project_root))
        test_env["PYTHONPATH"] = str(worktree_project_root)

        from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
        from codeflash.languages.python.test_runner import async_execute_test_subprocess

        pytest_cmd_list = opt.language_support.build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)  # type: ignore[attr-defined]

        blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]
        blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]

        result_file_path = get_run_tmp_file(Path(f"pytest_results_candidate_{candidate_index}_{slot.index}.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

        common_pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
            f"--timeout={INDIVIDUAL_TESTCASE_TIMEOUT}",
        ]

        cmd = pytest_cmd_list + common_pytest_args + blocklist_args + result_args + behavior_test_files

        try:
            behavior_result = await async_execute_test_subprocess(
                cmd_list=cmd, cwd=slot.path, env=pytest_test_env, timeout=600
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Behavioral test timeout for candidate {candidate_index}")
            return Failure(EvalFailure(message="Behavioral test timeout"))

        from codeflash.verification.parse_test_output import parse_test_xml

        behavior_test_results = parse_test_xml(
            result_file_path, test_files=opt.test_files, test_config=opt.test_cfg, run_result=behavior_result
        )

        if not behavior_test_results.test_results:
            return Failure(EvalFailure(message="No behavioral test results"))

        from codeflash.verification.equivalence import compare_test_results

        match, diffs = compare_test_results(
            original_code_baseline.behavior_test_results, behavior_test_results, pass_fail_only=True
        )

        if not match:
            return Failure(EvalFailure(message=f"Behavioral mismatch: {len(diffs)} diffs", diffs=diffs))

        return Success(
            _BehavioralPass(
                candidate_index=candidate_index,
                perf_test_files=perf_test_files,
                test_env=pytest_test_env,
                pytest_cmd_list=pytest_cmd_list,
                behavior_test_results=behavior_test_results,
            )
        )

    async def _benchmark_phase(
        self, slot: WorktreeSlot, bp: _BehavioralPass, original_code_baseline: OriginalCodeBaseline
    ) -> Result[OptimizedCandidateResult, EvalFailure]:
        """Run performance benchmarks sequentially for a candidate that passed behavioral tests."""
        opt = self._optimizer

        # Re-stage the candidate code in the acquired slot
        fto = opt.function_to_optimize
        for file in opt.test_files.test_files:
            if file.benchmarking_file_path and file.benchmarking_file_path.exists():
                await slot.write_candidate(
                    file.benchmarking_file_path, file.benchmarking_file_path.read_text(encoding="utf-8")
                )

        blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]
        blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]

        perf_result_file = get_run_tmp_file(Path(f"pytest_perf_candidate_{bp.candidate_index}_{slot.index}.xml"))
        perf_result_args = [f"--junitxml={perf_result_file.as_posix()}", "-o", "junit_logging=all"]

        perf_pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=5",
            "--codeflash_max_loops=250",
            f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
            f"--timeout={INDIVIDUAL_TESTCASE_TIMEOUT}",
        ]

        perf_cmd = bp.pytest_cmd_list + perf_pytest_args + blocklist_args + perf_result_args + bp.perf_test_files

        from codeflash.languages.python.test_runner import async_execute_test_subprocess

        try:
            await async_execute_test_subprocess(cmd_list=perf_cmd, cwd=slot.path, env=bp.test_env, timeout=600)
        except subprocess.TimeoutExpired:
            logger.warning(f"Performance test timeout for candidate {bp.candidate_index}")
            return Failure(EvalFailure(message="Performance test timeout"))

        from codeflash.verification.parse_test_output import parse_test_xml

        perf_test_results = parse_test_xml(perf_result_file, test_files=opt.test_files, test_config=opt.test_cfg)

        if not perf_test_results.test_results:
            return Failure(EvalFailure(message="No performance test results"))

        loop_count = perf_test_results.effective_loop_count()
        total_timing = perf_test_results.total_passed_runtime()

        if total_timing == 0:
            return Failure(EvalFailure(message="Zero runtime for optimized candidate"))

        from codeflash.models.models import OptimizedCandidateResult

        return Success(
            OptimizedCandidateResult(
                max_loop_count=loop_count,
                best_test_runtime=total_timing,
                behavior_test_results=bp.behavior_test_results,
                benchmarking_test_results=perf_test_results,
                replay_benchmarking_test_results=None,
                optimization_candidate_index=bp.candidate_index,
                total_candidate_timing=total_timing,
                async_throughput=None,
                concurrency_metrics=None,
            )
        )

    @staticmethod
    def _replace_and_capture(
        opt: FunctionOptimizer,
        code_context: CodeOptimizationContext,
        candidate: OptimizedCandidate,
        original_helper_code: dict[Path, str],
    ) -> tuple[str, dict[Path, str]] | None:
        """Apply code replacement to main tree, capture the result, restore original."""
        fto = opt.function_to_optimize
        try:
            did_update = opt.replace_function_and_helpers_with_optimized_code(
                code_context=code_context,
                optimized_code=candidate.source_code,
                original_helper_code=original_helper_code,
            )
            if not did_update:
                return None

            fto_code = Path(fto.file_path).read_text("utf-8")
            helper_codes = {Path(p): Path(p).read_text("utf-8") for p in original_helper_code}
            return fto_code, helper_codes
        except (ValueError, SyntaxError, AttributeError) as e:
            logger.error(f"Code replacement failed: {e}")
            return None
        finally:
            opt.write_code_and_helpers(opt.function_to_optimize_source_code, original_helper_code, fto.file_path)


def run_parallel_evaluation(
    optimizer: FunctionOptimizer,
    candidates: list[tuple[CandidateNode, int, str | None]],
    code_context: CodeOptimizationContext,
    original_code_baseline: OriginalCodeBaseline,
    original_helper_code: dict[Path, str],
    file_path_to_helper_classes: dict[Path, set[str]],
    eval_ctx: CandidateEvaluationContext,
    exp_type: str,
    pool_size: int = 4,
) -> tuple[list[tuple[CandidateNode, Result[OptimizedCandidateResult, EvalFailure] | None]], list, list]:
    """Entry point: run parallel candidate evaluation from sync code via anyio.

    Returns (eval_results, [], []).  The empty lists maintain backward compatibility.
    """
    evaluator = ParallelCandidateEvaluator(optimizer, pool_size=pool_size)

    async def _run() -> list[tuple[CandidateNode, Result[OptimizedCandidateResult, EvalFailure] | None]]:
        return await evaluator.evaluate_candidates(
            candidates=candidates,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
        )

    results = anyio.run(_run)
    return results, [], []
