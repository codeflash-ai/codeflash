"""Java concurrency pattern detection and analysis.

This module provides functionality to detect and analyze concurrent patterns
in Java code, including:
- CompletableFuture usage
- Parallel streams
- ExecutorService and thread pools
- Virtual threads (Java 21+)
- Synchronized methods/blocks
- Concurrent collections
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.languages.base import FunctionInfo

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyInfo:
    """Information about concurrency in a function."""

    is_concurrent: bool
    """Whether the function uses concurrent patterns."""

    patterns: list[str]
    """List of concurrent patterns detected (e.g., 'CompletableFuture', 'parallel_stream')."""

    has_completable_future: bool = False
    """Uses CompletableFuture."""

    has_parallel_stream: bool = False
    """Uses parallel streams."""

    has_executor_service: bool = False
    """Uses ExecutorService or thread pools."""

    has_virtual_threads: bool = False
    """Uses virtual threads (Java 21+)."""

    has_synchronized: bool = False
    """Has synchronized methods or blocks."""

    has_concurrent_collections: bool = False
    """Uses concurrent collections (ConcurrentHashMap, etc.)."""

    has_atomic_operations: bool = False
    """Uses atomic operations (AtomicInteger, etc.)."""

    async_method_calls: list[str] = None
    """List of async/concurrent method calls."""

    def __post_init__(self):
        if self.async_method_calls is None:
            self.async_method_calls = []


class JavaConcurrencyAnalyzer:
    """Analyzes Java code for concurrency patterns."""

    # Concurrent patterns to detect
    COMPLETABLE_FUTURE_PATTERNS = {
        "CompletableFuture",
        "supplyAsync",
        "runAsync",
        "thenApply",
        "thenAccept",
        "thenCompose",
        "thenCombine",
        "allOf",
        "anyOf",
    }

    EXECUTOR_PATTERNS = {
        "ExecutorService",
        "Executors",
        "ThreadPoolExecutor",
        "ScheduledExecutorService",
        "ForkJoinPool",
        "newCachedThreadPool",
        "newFixedThreadPool",
        "newSingleThreadExecutor",
        "newScheduledThreadPool",
        "newWorkStealingPool",
    }

    VIRTUAL_THREAD_PATTERNS = {
        "newVirtualThreadPerTaskExecutor",
        "Thread.startVirtualThread",
        "Thread.ofVirtual",
        "VirtualThreads",
    }

    CONCURRENT_COLLECTION_PATTERNS = {
        "ConcurrentHashMap",
        "ConcurrentLinkedQueue",
        "ConcurrentLinkedDeque",
        "ConcurrentSkipListMap",
        "ConcurrentSkipListSet",
        "CopyOnWriteArrayList",
        "CopyOnWriteArraySet",
        "BlockingQueue",
        "LinkedBlockingQueue",
        "ArrayBlockingQueue",
    }

    ATOMIC_PATTERNS = {
        "AtomicInteger",
        "AtomicLong",
        "AtomicBoolean",
        "AtomicReference",
        "AtomicIntegerArray",
        "AtomicLongArray",
        "AtomicReferenceArray",
    }

    def __init__(self, analyzer=None):
        """Initialize concurrency analyzer.

        Args:
            analyzer: Optional JavaAnalyzer for parsing.

        """
        self.analyzer = analyzer

    def analyze_function(self, func: FunctionInfo, source: str | None = None) -> ConcurrencyInfo:
        """Analyze a function for concurrency patterns.

        Args:
            func: Function to analyze.
            source: Optional source code (if not provided, will read from file).

        Returns:
            ConcurrencyInfo with detected patterns.

        """
        if source is None:
            try:
                source = func.file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read source for %s: %s", func.name, e)
                return ConcurrencyInfo(is_concurrent=False, patterns=[])

        # Extract function source
        lines = source.splitlines()
        func_start = func.start_line - 1  # Convert to 0-indexed
        func_end = func.end_line
        func_source = "\n".join(lines[func_start:func_end])

        # Detect patterns
        patterns = []
        has_completable_future = False
        has_parallel_stream = False
        has_executor_service = False
        has_virtual_threads = False
        has_synchronized = False
        has_concurrent_collections = False
        has_atomic_operations = False
        async_method_calls = []

        # Check for CompletableFuture
        for pattern in self.COMPLETABLE_FUTURE_PATTERNS:
            if pattern in func_source:
                has_completable_future = True
                patterns.append(f"CompletableFuture.{pattern}")
                async_method_calls.append(pattern)

        # Check for parallel streams
        if ".parallel()" in func_source or ".parallelStream()" in func_source:
            has_parallel_stream = True
            patterns.append("parallel_stream")
            async_method_calls.append("parallel")

        # Check for ExecutorService
        for pattern in self.EXECUTOR_PATTERNS:
            if pattern in func_source:
                has_executor_service = True
                patterns.append(f"Executor.{pattern}")
                async_method_calls.append(pattern)

        # Check for virtual threads (Java 21+)
        for pattern in self.VIRTUAL_THREAD_PATTERNS:
            if pattern in func_source:
                has_virtual_threads = True
                patterns.append(f"VirtualThread.{pattern}")
                async_method_calls.append(pattern)

        # Check for synchronized
        if "synchronized" in func_source:
            has_synchronized = True
            patterns.append("synchronized")

        # Check for concurrent collections
        for pattern in self.CONCURRENT_COLLECTION_PATTERNS:
            if pattern in func_source:
                has_concurrent_collections = True
                patterns.append(f"ConcurrentCollection.{pattern}")

        # Check for atomic operations
        for pattern in self.ATOMIC_PATTERNS:
            if pattern in func_source:
                has_atomic_operations = True
                patterns.append(f"Atomic.{pattern}")

        is_concurrent = bool(patterns)

        return ConcurrencyInfo(
            is_concurrent=is_concurrent,
            patterns=patterns,
            has_completable_future=has_completable_future,
            has_parallel_stream=has_parallel_stream,
            has_executor_service=has_executor_service,
            has_virtual_threads=has_virtual_threads,
            has_synchronized=has_synchronized,
            has_concurrent_collections=has_concurrent_collections,
            has_atomic_operations=has_atomic_operations,
            async_method_calls=async_method_calls,
        )

    def analyze_source(self, source: str, file_path: Path | None = None) -> dict[str, ConcurrencyInfo]:
        """Analyze entire source file for concurrency patterns.

        Args:
            source: Java source code.
            file_path: Optional file path for context.

        Returns:
            Dictionary mapping function names to their ConcurrencyInfo.

        """
        # This would require parsing the source to extract all functions
        # For now, return empty dict - can be implemented later if needed
        return {}

    @staticmethod
    def should_measure_throughput(concurrency_info: ConcurrencyInfo) -> bool:
        """Determine if throughput should be measured for concurrent code.

        Args:
            concurrency_info: Concurrency information for a function.

        Returns:
            True if throughput measurement is recommended.

        """
        # Measure throughput for async patterns that execute multiple operations
        return (
            concurrency_info.has_completable_future
            or concurrency_info.has_parallel_stream
            or concurrency_info.has_executor_service
            or concurrency_info.has_virtual_threads
        )

    @staticmethod
    def get_optimization_suggestions(concurrency_info: ConcurrencyInfo) -> list[str]:
        """Get optimization suggestions based on detected patterns.

        Args:
            concurrency_info: Concurrency information for a function.

        Returns:
            List of optimization suggestions.

        """
        suggestions = []

        if concurrency_info.has_completable_future:
            suggestions.append(
                "Consider using CompletableFuture.allOf() or thenCompose() "
                "to combine multiple async operations efficiently"
            )

        if concurrency_info.has_parallel_stream:
            suggestions.append(
                "Parallel streams work best with CPU-bound tasks. "
                "For I/O-bound tasks, consider CompletableFuture or virtual threads"
            )

        if concurrency_info.has_executor_service and concurrency_info.has_virtual_threads:
            suggestions.append(
                "You're using both traditional thread pools and virtual threads. "
                "Consider migrating fully to virtual threads for better resource utilization"
            )

        if not concurrency_info.has_concurrent_collections and concurrency_info.is_concurrent:
            suggestions.append(
                "Consider using concurrent collections (ConcurrentHashMap, etc.) "
                "instead of synchronized collections for better performance"
            )

        if not concurrency_info.has_atomic_operations and concurrency_info.has_synchronized:
            suggestions.append(
                "Consider using atomic operations (AtomicInteger, etc.) "
                "instead of synchronized blocks for simple counters"
            )

        return suggestions


def analyze_function_concurrency(func: FunctionInfo, source: str | None = None, analyzer=None) -> ConcurrencyInfo:
    """Analyze a function for concurrency patterns.

    Convenience function that creates a JavaConcurrencyAnalyzer and analyzes the function.

    Args:
        func: Function to analyze.
        source: Optional source code.
        analyzer: Optional JavaAnalyzer.

    Returns:
        ConcurrencyInfo with detected patterns.

    """
    concurrency_analyzer = JavaConcurrencyAnalyzer(analyzer)
    return concurrency_analyzer.analyze_function(func, source)
