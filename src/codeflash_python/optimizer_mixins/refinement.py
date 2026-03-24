from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_python.code_utils.config_consts import MIN_CORRECT_CANDIDATES, EffortKeys, get_effort_value
from codeflash_python.models.models import OptimizedCandidateSource

if TYPE_CHECKING:
    import concurrent.futures

    from codeflash_python.api.aiservice import AiServiceClient
    from codeflash_python.api.types import TestDiff
    from codeflash_python.models.models import CodeOptimizationContext, OptimizedCandidate
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
    from codeflash_python.optimizer_mixins.candidate_structures import CandidateEvaluationContext
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class RefinementMixin(_Base):
    def call_adaptive_optimize(
        self,
        trace_id: str,
        original_source_code: str,
        prev_candidates: list[OptimizedCandidate],
        eval_ctx: CandidateEvaluationContext,
        ai_service_client: AiServiceClient,
    ) -> concurrent.futures.Future[OptimizedCandidate | None] | None:
        if self.adaptive_optimization_counter >= get_effort_value(
            EffortKeys.MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE, self.effort
        ):
            logger.debug(
                "Max adaptive optimizations reached for %s: %s",
                self.function_to_optimize.qualified_name,
                self.adaptive_optimization_counter,
            )
            return None

        adaptive_count = sum(1 for c in prev_candidates if c.source == OptimizedCandidateSource.ADAPTIVE)

        if adaptive_count >= get_effort_value(EffortKeys.ADAPTIVE_OPTIMIZATION_THRESHOLD, self.effort):
            return None

        from codeflash_python.api.types import AdaptiveOptimizedCandidate, AIServiceAdaptiveOptimizeRequest

        request_candidates = []

        for c in prev_candidates:
            speedup = eval_ctx.get_speedup_ratio(c.optimization_id)
            request_candidates.append(
                AdaptiveOptimizedCandidate(
                    optimization_id=c.optimization_id,
                    source_code=c.source_code.markdown,
                    explanation=c.explanation,
                    source=c.source,
                    speedup=f"Performance gain: {int(speedup * 100 + 0.5)}%"
                    if speedup
                    else "Candidate didn't match the behavior of the original code",
                )
            )

        request = AIServiceAdaptiveOptimizeRequest(
            trace_id=trace_id, original_source_code=original_source_code, candidates=request_candidates
        )
        self.adaptive_optimization_counter += 1
        return self.executor.submit(ai_service_client.adaptive_optimize, request=request)

    def repair_optimization(
        self,
        original_source_code: str,
        modified_source_code: str,
        test_diffs: list[TestDiff],
        trace_id: str,
        optimization_id: str,
        ai_service_client: AiServiceClient,
        executor: concurrent.futures.ThreadPoolExecutor,
        language: str = "python",
    ) -> concurrent.futures.Future[OptimizedCandidate | None]:
        from codeflash_python.api.types import AIServiceCodeRepairRequest

        request = AIServiceCodeRepairRequest(
            optimization_id=optimization_id,
            original_source_code=original_source_code,
            modified_source_code=modified_source_code,
            test_diffs=test_diffs,
            trace_id=trace_id,
            language=language,
        )
        return executor.submit(ai_service_client.code_repair, request=request)

    def repair_if_possible(
        self,
        candidate: OptimizedCandidate,
        diffs: list[TestDiff],
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        test_results_count: int,
        exp_type: str,
    ) -> None:
        max_repairs = get_effort_value(EffortKeys.MAX_CODE_REPAIRS_PER_TRACE, self.effort)
        if self.repair_counter >= max_repairs:
            logger.debug("Repair counter reached %s, skipping repair", max_repairs)
            return

        successful_candidates_count = sum(1 for is_correct in eval_ctx.is_correct.values() if is_correct)
        if successful_candidates_count >= MIN_CORRECT_CANDIDATES:
            logger.debug("%s of the candidates were correct, no need to repair", successful_candidates_count)
            return

        if candidate.source not in (OptimizedCandidateSource.OPTIMIZE, OptimizedCandidateSource.OPTIMIZE_LP):
            # only repair the first pass of the candidates for now
            logger.debug("Candidate is a result of %s, skipping repair", candidate.source.value)
            return
        if not diffs:
            logger.debug("No diffs found, skipping repair")
            return
        result_unmatched_perc = len(diffs) / test_results_count
        if result_unmatched_perc > get_effort_value(EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT, self.effort):
            logger.debug("Result unmatched percentage is %s%%, skipping repair", result_unmatched_perc * 100)
            return

        logger.debug(
            "Adding a candidate for repair, with %s diffs, (%s%% unmatched)", len(diffs), result_unmatched_perc * 100
        )
        # start repairing
        ai_service_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
        assert ai_service_client is not None
        self.repair_counter += 1
        self.future_all_code_repair.append(
            self.repair_optimization(
                original_source_code=code_context.read_writable_code.markdown,
                modified_source_code=candidate.source_code.markdown,
                test_diffs=diffs,
                trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
                ai_service_client=ai_service_client,
                optimization_id=candidate.optimization_id,
                executor=self.executor,
                language=self.function_to_optimize.language,
            )
        )
