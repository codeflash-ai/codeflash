"""Mixin: AI candidate generation, repair, refinement, adaptive optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_core.models import Candidate
from codeflash_python.plugin_helpers import format_speedup_pct, map_candidate_source

if TYPE_CHECKING:
    from codeflash_core.models import BenchmarkResults, CodeContext, ScoredCandidate, TestDiff
    from codeflash_python.plugin import PythonPlugin as _Base
else:
    _Base = object

logger = logging.getLogger(__name__)


class PluginAiOpsMixin(_Base):  # type: ignore[cyclic-class-definition]
    def get_candidates(self, context: CodeContext, trace_id: str = "") -> list[Candidate]:
        client = self.get_ai_client()
        assert trace_id, "trace_id must be provided"

        # Use cached internal context for markdown-formatted code (what the API expects)
        internal_ctx = self.last_internal_context
        if internal_ctx is not None:
            source_code = internal_ctx.read_writable_code.markdown
            dependency_code = internal_ctx.read_only_context_code
        else:
            source_code = context.target_code
            dependency_code = context.read_only_context

        optimized = client.optimize_code(
            source_code=source_code,
            dependency_code=dependency_code,
            trace_id=trace_id,
            language="python",
            is_numerical_code=self.is_numerical_code,
        )

        candidates = []
        for opt in optimized:
            code = opt.source_code.flat if opt.source_code else ""
            code_md = opt.source_code.markdown if opt.source_code else ""
            if code:
                candidates.append(
                    Candidate(code=code, explanation=opt.explanation or "", source="optimize", code_markdown=code_md)
                )
        return candidates

    def get_line_profiler_candidates(
        self, context: CodeContext, line_profile_data: str, trace_id: str = ""
    ) -> list[Candidate]:
        assert trace_id, "trace_id must be provided"
        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for line profiler")
            return []

        internal_ctx = self.last_internal_context
        source_code = internal_ctx.read_writable_code.markdown if internal_ctx else context.target_code
        dependency_code = internal_ctx.read_only_context_code if internal_ctx else context.read_only_context

        optimized = client.optimize_python_code_line_profiler(
            source_code=source_code,
            dependency_code=dependency_code,
            trace_id=trace_id,
            line_profiler_results=line_profile_data,
            n_candidates=3,
        )

        candidates = []
        for opt in optimized:
            code = opt.source_code.flat if opt.source_code else ""
            code_md = opt.source_code.markdown if opt.source_code else ""
            if code:
                candidates.append(
                    Candidate(
                        code=code, explanation=opt.explanation or "", source="line_profiler", code_markdown=code_md
                    )
                )
        return candidates

    def repair_candidate(
        self, context: CodeContext, candidate: Candidate, test_diffs: list[TestDiff], trace_id: str = ""
    ) -> Candidate | None:
        assert trace_id, "trace_id must be provided"
        from codeflash_python.api.types import AIServiceCodeRepairRequest, TestDiffScope
        from codeflash_python.api.types import TestDiff as InternalTestDiff

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for repair")
            return None

        internal_ctx = self.last_internal_context
        source_code = internal_ctx.read_writable_code.markdown if internal_ctx else context.target_code
        modified_code = candidate.code_markdown or candidate.code

        internal_diffs = [
            InternalTestDiff(
                scope=TestDiffScope.RETURN_VALUE,
                original_pass=True,
                candidate_pass=False,
                original_value=str(d.baseline_output) if d.baseline_output is not None else None,
                candidate_value=str(d.candidate_output) if d.candidate_output is not None else None,
            )
            for d in test_diffs
        ]

        request = AIServiceCodeRepairRequest(
            optimization_id=candidate.candidate_id,
            original_source_code=source_code,
            modified_source_code=modified_code,
            trace_id=trace_id,
            test_diffs=internal_diffs,
        )

        try:
            result = client.code_repair(request)
        except Exception:
            logger.exception("Code repair API call failed")
            return None

        if result is None:
            return None

        code = result.source_code.flat if result.source_code else ""
        code_md = result.source_code.markdown if result.source_code else ""
        if not code:
            return None

        return Candidate(
            code=code,
            explanation=result.explanation or "",
            source="repair",
            parent_id=candidate.candidate_id,
            code_markdown=code_md,
        )

    def refine_candidate(
        self, context: CodeContext, candidate: ScoredCandidate, baseline_bench: BenchmarkResults, trace_id: str = ""
    ) -> list[Candidate]:
        assert trace_id, "trace_id must be provided"
        from codeflash_python.api.types import AIServiceRefinerRequest

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for refinement")
            return []

        internal_ctx = self.last_internal_context
        source_code = internal_ctx.read_writable_code.markdown if internal_ctx else context.target_code
        dependency_code = internal_ctx.read_only_context_code if internal_ctx else context.read_only_context
        optimized_code = candidate.candidate.code_markdown or candidate.candidate.code

        request = AIServiceRefinerRequest(
            optimization_id=candidate.candidate.candidate_id,
            original_source_code=source_code,
            read_only_dependency_code=dependency_code,
            original_code_runtime=int(baseline_bench.total_time * 1e9),
            optimized_source_code=optimized_code,
            optimized_explanation=candidate.candidate.explanation,
            optimized_code_runtime=int(candidate.benchmark_results.total_time * 1e9),
            speedup=format_speedup_pct(candidate.speedup),
            trace_id=trace_id,
            original_line_profiler_results="",
            optimized_line_profiler_results="",
        )

        try:
            results = client.optimize_code_refinement([request])
        except Exception:
            logger.exception("Code refinement API call failed")
            return []

        candidates = []
        for opt in results:
            code = opt.source_code.flat if opt.source_code else ""
            code_md = opt.source_code.markdown if opt.source_code else ""
            if code:
                candidates.append(
                    Candidate(
                        code=code,
                        explanation=opt.explanation or "",
                        source="refine",
                        parent_id=candidate.candidate.candidate_id,
                        code_markdown=code_md,
                    )
                )
        return candidates

    def adaptive_optimize(
        self, context: CodeContext, scored: list[ScoredCandidate], trace_id: str = ""
    ) -> Candidate | None:
        assert trace_id, "trace_id must be provided"
        from codeflash_python.api.types import AdaptiveOptimizedCandidate, AIServiceAdaptiveOptimizeRequest

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for adaptive optimization")
            return None

        internal_ctx = self.last_internal_context
        source_code = internal_ctx.read_writable_code.flat if internal_ctx else context.target_code

        adaptive_candidates = [
            AdaptiveOptimizedCandidate(
                optimization_id=sc.candidate.candidate_id,
                source_code=sc.candidate.code,
                explanation=sc.candidate.explanation,
                source=map_candidate_source(sc.candidate.source),
                speedup=f"Performance gain: {int(sc.speedup * 100 + 0.5)}%"
                if sc.speedup > 0
                else "Candidate didn't match the behavior of the original code",
            )
            for sc in scored
        ]

        request = AIServiceAdaptiveOptimizeRequest(
            trace_id=trace_id, original_source_code=source_code, candidates=adaptive_candidates
        )

        try:
            result = client.adaptive_optimize(request)
        except Exception:
            logger.exception("Adaptive optimization API call failed")
            return None

        if result is None:
            return None

        code = result.source_code.flat if result.source_code else ""
        if not code:
            return None

        return Candidate(code=code, explanation=result.explanation or "", source="adaptive")
