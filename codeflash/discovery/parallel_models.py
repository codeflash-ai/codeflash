"""Data models and classes for parallel test discovery infrastructure."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from codeflash.models.models import FunctionCalledInTest, TestsInFile


@dataclass(frozen=True)
class ImportAnalysisTask:
    """Task for analyzing imports in a test file to check for target functions."""
    
    test_file_path: Path
    target_functions: Set[str]


@dataclass(frozen=True)
class ImportAnalysisResult:
    """Result of import analysis for a test file."""
    
    test_file_path: Path
    has_target_imports: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class JediAnalysisTask:
    """Task for Jedi-based analysis of test files to find function references."""
    
    test_file: Path
    test_functions: List[TestsInFile]
    project_root: Path
    test_framework: str


@dataclass(frozen=True)
class JediAnalysisResult:
    """Result of Jedi analysis for a test file."""
    
    test_file: Path
    function_mappings: Dict[str, Set[FunctionCalledInTest]]
    test_count: int
    replay_test_count: int
    error: Optional[str] = None


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel test discovery processing."""
    
    max_workers: Optional[int] = None  # Auto-detect if None
    enable_parallel: bool = True
    chunk_size: int = 1  # Files per task
    timeout_seconds: int = 300  # Per-file timeout
    fallback_on_error: bool = True  # Fall back to sequential on errors


@dataclass
class ProgressInfo:
    """Progress tracking information for parallel processing."""
    
    total_files: int
    completed_files: int
    failed_files: int
    current_phase: str  # "import_analysis" or "jedi_processing"
    start_time: float
    estimated_remaining: Optional[float] = None


class ErrorAction(enum.Enum):
    """Actions to take when encountering errors during parallel processing."""
    
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK_TO_SEQUENTIAL = "fallback_to_sequential"
    ABORT = "abort"


@dataclass
class ProcessingError:
    """Information about an error that occurred during processing."""
    
    task_id: str
    error_type: str
    error_message: str
    retry_count: int
    timestamp: float