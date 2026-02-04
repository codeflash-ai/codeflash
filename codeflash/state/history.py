from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.state.models import OptimizationAttempt, OptimizationStatus

if TYPE_CHECKING:
    from codeflash.state.store import StateStore


class OptimizationHistory:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def record_attempt(self, attempt: OptimizationAttempt) -> None:
        self.store.persist_optimization_attempt(attempt)

    def get_function_attempts(self, qualified_name: str, limit: int = 100) -> list[OptimizationAttempt]:
        return self.store.get_function_history(qualified_name, limit=limit)

    def get_successful_optimizations(
        self, qualified_name: str | None = None, limit: int = 100
    ) -> list[OptimizationAttempt]:
        if qualified_name:
            attempts = self.store.get_function_history(qualified_name, limit=limit)
            return [a for a in attempts if a.status == OptimizationStatus.COMPLETED]
        return self.store.get_recent_attempts(status=OptimizationStatus.COMPLETED, limit=limit)

    def get_failed_optimizations(self, qualified_name: str | None = None, limit: int = 100) -> list[OptimizationAttempt]:
        if qualified_name:
            attempts = self.store.get_function_history(qualified_name, limit=limit)
            return [a for a in attempts if a.status == OptimizationStatus.FAILED]
        return self.store.get_recent_attempts(status=OptimizationStatus.FAILED, limit=limit)

    def should_skip_function(self, qualified_name: str, code_hash: str | None = None) -> tuple[bool, str | None]:
        if code_hash and self.store.was_function_recently_optimized(qualified_name, code_hash=code_hash):
            return True, "Function with same code hash was recently optimized"

        recent_attempts = self.store.get_function_history(qualified_name, limit=5)

        completed_count = sum(1 for a in recent_attempts if a.status == OptimizationStatus.COMPLETED)
        if completed_count >= 3:
            return True, "Function has been successfully optimized multiple times recently"

        failed_count = sum(1 for a in recent_attempts if a.status == OptimizationStatus.FAILED)
        if failed_count >= 3:
            return True, "Function has failed optimization multiple times recently"

        return False, None

    def get_best_speedup(self, qualified_name: str) -> float | None:
        attempts = self.store.get_function_history(qualified_name)
        speedups = [a.speedup for a in attempts if a.speedup is not None and a.status == OptimizationStatus.COMPLETED]
        return max(speedups) if speedups else None

    def get_statistics(self) -> dict[str, int]:
        recent = self.store.get_recent_attempts(limit=1000)
        return {
            "total_attempts": len(recent),
            "completed": sum(1 for a in recent if a.status == OptimizationStatus.COMPLETED),
            "failed": sum(1 for a in recent if a.status == OptimizationStatus.FAILED),
            "skipped": sum(1 for a in recent if a.status == OptimizationStatus.SKIPPED),
            "in_progress": sum(1 for a in recent if a.status == OptimizationStatus.IN_PROGRESS),
        }

    def cleanup(self, days: int = 30) -> int:
        return self.store.cleanup_old_records(days=days)
