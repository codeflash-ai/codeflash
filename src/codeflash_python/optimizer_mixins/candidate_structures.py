from __future__ import annotations

import concurrent.futures
import logging
import queue
from typing import TYPE_CHECKING, Callable

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeflash_python.code_utils.config_consts import REFINED_CANDIDATE_RANKING_WEIGHTS, EffortKeys, get_effort_value
from codeflash_python.optimizer_mixins.scoring import (
    choose_weights,
    create_score_dictionary_from_metrics,
    diff_length,
    normalize_by_max,
)

if TYPE_CHECKING:
    from codeflash_python.models.models import CodeOptimizationContext, OptimizedCandidate

logger = logging.getLogger("codeflash_python")


class CandidateNode:
    __slots__ = ("candidate", "children", "parent")

    def __init__(self, candidate: OptimizedCandidate) -> None:
        self.candidate = candidate
        self.parent: CandidateNode | None = None
        self.children: list[CandidateNode] = []

    def is_leaf(self) -> bool:
        return not self.children

    def path_to_root(self) -> list[OptimizedCandidate]:
        path = []
        node: CandidateNode | None = self
        while node:
            path.append(node.candidate)
            node = node.parent
        return path[::-1]


class CandidateForest:
    def __init__(self) -> None:
        self.nodes: dict[str, CandidateNode] = {}

    def add(self, candidate: OptimizedCandidate) -> CandidateNode:
        cid = candidate.optimization_id
        pid = candidate.parent_id

        node = self.nodes.get(cid)
        if node is None:
            node = CandidateNode(candidate)
            self.nodes[cid] = node

        if pid is not None:
            parent = self.nodes.get(pid)
            if parent is None:
                parent = CandidateNode(candidate=None)  # type: ignore[arg-type]  # placeholder
                self.nodes[pid] = parent

            node.parent = parent
            parent.children.append(node)

        return node

    def get_node(self, cid: str) -> CandidateNode | None:
        return self.nodes.get(cid)


class CandidateProcessor:
    """Handles candidate processing using a queue-based approach."""

    def __init__(
        self,
        initial_candidates: list[OptimizedCandidate],
        future_line_profile_results: concurrent.futures.Future,
        eval_ctx: CandidateEvaluationContext,
        effort: str,
        original_markdown_code: str,
        future_all_refinements: list[concurrent.futures.Future],
        future_all_code_repair: list[concurrent.futures.Future],
        future_adaptive_optimizations: list[concurrent.futures.Future],
    ) -> None:
        self.candidate_queue = queue.Queue()
        self.forest = CandidateForest()
        self.line_profiler_done = False
        self.refinement_done = False
        self.eval_ctx = eval_ctx
        self.effort = effort
        self.candidate_len = len(initial_candidates)
        self.refinement_calls_count = 0
        self.original_markdown_code = original_markdown_code

        # Initialize queue with initial candidates
        for candidate in initial_candidates:
            self.forest.add(candidate)
            self.candidate_queue.put(candidate)

        self.future_line_profile_results = future_line_profile_results
        self.future_all_refinements = future_all_refinements
        self.future_all_code_repair = future_all_code_repair
        self.future_adaptive_optimizations = future_adaptive_optimizations

    def get_total_llm_calls(self) -> int:
        return self.refinement_calls_count

    def get_next_candidate(self) -> CandidateNode | None:
        """Get the next candidate from the queue, handling async results as needed."""
        try:
            return self.forest.get_node(self.candidate_queue.get_nowait().optimization_id)
        except queue.Empty:
            return self.handle_empty_queue()

    def handle_empty_queue(self) -> CandidateNode | None:
        """Handle empty queue by checking for pending async results."""
        if not self.line_profiler_done:
            return self.process_candidates(
                [self.future_line_profile_results],
                "all candidates processed, await candidates from line profiler",
                "Added results from line profiler to candidates, total candidates now: {1}",
                lambda: setattr(self, "line_profiler_done", True),
            )
        if len(self.future_all_code_repair) > 0:
            return self.process_candidates(
                self.future_all_code_repair,
                "Repairing {0} candidates",
                "Added {0} candidates from repair, total candidates now: {1}",
                self.future_all_code_repair.clear,
            )
        if self.line_profiler_done and not self.refinement_done:
            return self.process_candidates(
                self.future_all_refinements,
                "Refining generated code for improved quality and performance...",
                "Added {0} candidates from refinement, total candidates now: {1}",
                lambda: setattr(self, "refinement_done", True),
                filter_candidates_func=self.filter_refined_candidates,
            )
        if len(self.future_adaptive_optimizations) > 0:
            return self.process_candidates(
                self.future_adaptive_optimizations,
                "Applying adaptive optimizations to {0} candidates",
                "Added {0} candidates from adaptive optimization, total candidates now: {1}",
                self.future_adaptive_optimizations.clear,
            )
        return None  # All done

    def process_candidates(
        self,
        future_candidates: list[concurrent.futures.Future],
        loading_msg: str,
        success_msg: str,
        callback: Callable[[], None],
        filter_candidates_func: Callable[[list[OptimizedCandidate]], list[OptimizedCandidate]] | None = None,
    ) -> CandidateNode | None:
        if len(future_candidates) == 0:
            return None
        concurrent.futures.wait(future_candidates)
        candidates: list[OptimizedCandidate] = []
        for future_c in future_candidates:
            candidate_result = future_c.result()
            if not candidate_result:
                continue

            if isinstance(candidate_result, list):
                candidates.extend(candidate_result)
            else:
                candidates.append(candidate_result)

        candidates = filter_candidates_func(candidates) if filter_candidates_func else candidates
        for candidate in candidates:
            self.forest.add(candidate)
            self.candidate_queue.put(candidate)
            self.candidate_len += 1

        if len(candidates) > 0:
            logger.info(success_msg.format(len(candidates), self.candidate_len))

        callback()
        return self.get_next_candidate()

    def filter_refined_candidates(self, candidates: list[OptimizedCandidate]) -> list[OptimizedCandidate]:
        """We generate a weighted ranking based on the runtime and diff lines and select the best of valid optimizations to be tested."""
        self.refinement_calls_count += len(candidates)

        top_n_candidates = int(
            min(int(get_effort_value(EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT, self.effort)), len(candidates))
        )

        if len(candidates) == top_n_candidates:
            # no need for ranking since we will return all candidates
            return candidates

        diff_lens_list = []
        runtimes_list = []
        for c in candidates:
            # current refined candidates is not benchmarked yet, a close values we would expect to be the parent candidate
            parent_id = c.parent_id
            if parent_id is None:
                continue
            parent_candidate_node = self.forest.get_node(parent_id)
            parent_optimized_runtime = self.eval_ctx.get_optimized_runtime(parent_id)
            if not parent_optimized_runtime or not parent_candidate_node:
                continue
            diff_lens_list.append(
                diff_length(self.original_markdown_code, parent_candidate_node.candidate.source_code.markdown)
            )
            runtimes_list.append(parent_optimized_runtime)

        if not runtimes_list or not diff_lens_list:
            # should not happen
            logger.warning("No valid candidates for refinement while filtering")
            return candidates

        runtime_w, diff_w = REFINED_CANDIDATE_RANKING_WEIGHTS
        weights = choose_weights(runtime=runtime_w, diff=diff_w)

        runtime_norm = normalize_by_max(runtimes_list)
        diffs_norm = normalize_by_max(diff_lens_list)
        # the lower the better
        score_dict = create_score_dictionary_from_metrics(weights, runtime_norm, diffs_norm)
        top_indecies = sorted(score_dict, key=score_dict.get)[:top_n_candidates]  # type: ignore[arg-type]

        return [candidates[idx] for idx in top_indecies]

    def is_done(self) -> bool:
        """Check if processing is complete."""
        return (
            self.line_profiler_done
            and self.refinement_done
            and len(self.future_all_code_repair) == 0
            and len(self.future_adaptive_optimizations) == 0
            and self.candidate_queue.empty()
        )


@dataclass
class CandidateEvaluationContext:
    """Holds tracking state during candidate evaluation in determine_best_candidate."""

    speedup_ratios: dict[str, float | None] = Field(default_factory=dict)
    optimized_runtimes: dict[str, float | None] = Field(default_factory=dict)
    is_correct: dict[str, bool] = Field(default_factory=dict)
    optimized_line_profiler_results: dict[str, str] = Field(default_factory=dict)
    ast_code_to_id: dict = Field(default_factory=dict)
    optimizations_post: dict[str, str] = Field(default_factory=dict)
    valid_optimizations: list = Field(default_factory=list)

    def record_failed_candidate(self, optimization_id: str) -> None:
        """Record results for a failed candidate."""
        self.optimized_runtimes[optimization_id] = None
        self.is_correct[optimization_id] = False
        self.speedup_ratios[optimization_id] = None

    def record_successful_candidate(self, optimization_id: str, runtime: float, speedup: float) -> None:
        """Record results for a successful candidate."""
        self.optimized_runtimes[optimization_id] = runtime
        self.is_correct[optimization_id] = True
        self.speedup_ratios[optimization_id] = speedup

    def record_line_profiler_result(self, optimization_id: str, result: str) -> None:
        """Record line profiler results for a candidate."""
        self.optimized_line_profiler_results[optimization_id] = result

    def handle_duplicate_candidate(
        self, candidate: OptimizedCandidate, normalized_code: str, code_context: CodeOptimizationContext
    ) -> None:
        """Handle a candidate that has been seen before."""
        past_opt_id = self.ast_code_to_id[normalized_code]["optimization_id"]

        # Copy results from the previous evaluation
        self.speedup_ratios[candidate.optimization_id] = self.speedup_ratios[past_opt_id]
        self.is_correct[candidate.optimization_id] = self.is_correct[past_opt_id]
        self.optimized_runtimes[candidate.optimization_id] = self.optimized_runtimes[past_opt_id]

        # Line profiler results only available for successful runs
        if past_opt_id in self.optimized_line_profiler_results:
            self.optimized_line_profiler_results[candidate.optimization_id] = self.optimized_line_profiler_results[
                past_opt_id
            ]

        self.optimizations_post[candidate.optimization_id] = self.ast_code_to_id[normalized_code][
            "shorter_source_code"
        ].markdown
        self.optimizations_post[past_opt_id] = self.ast_code_to_id[normalized_code]["shorter_source_code"].markdown

        # Update to shorter code if this candidate has a shorter diff
        new_diff_len = diff_length(candidate.source_code.flat, code_context.read_writable_code.flat)
        if new_diff_len < self.ast_code_to_id[normalized_code]["diff_len"]:
            self.ast_code_to_id[normalized_code]["shorter_source_code"] = candidate.source_code
            self.ast_code_to_id[normalized_code]["diff_len"] = new_diff_len

    def register_new_candidate(
        self, normalized_code: str, candidate: OptimizedCandidate, code_context: CodeOptimizationContext
    ) -> None:
        """Register a new candidate that hasn't been seen before."""
        self.ast_code_to_id[normalized_code] = {
            "optimization_id": candidate.optimization_id,
            "shorter_source_code": candidate.source_code,
            "diff_len": diff_length(candidate.source_code.flat, code_context.read_writable_code.flat),
        }

    def get_speedup_ratio(self, optimization_id: str) -> float | None:
        return self.speedup_ratios.get(optimization_id)

    def get_optimized_runtime(self, optimization_id: str) -> float | None:
        return self.optimized_runtimes.get(optimization_id)
