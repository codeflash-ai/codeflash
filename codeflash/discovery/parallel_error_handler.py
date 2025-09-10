"""Error handling framework for parallel test discovery with retry and fallback logic."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from codeflash.cli_cmds.console import logger
from codeflash.discovery.parallel_models import ErrorAction, ProcessingError


class ParallelErrorHandler:
    """Handles errors during parallel processing with retry and fallback logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        fallback_threshold: int = 5,
        timeout_threshold: int = 3
    ):
        """Initialize error handler with configuration.
        
        Args:
            max_retries: Maximum number of retries per task
            fallback_threshold: Number of errors before falling back to sequential
            timeout_threshold: Number of timeouts before reducing worker count
        """
        self.max_retries = max_retries
        self.fallback_threshold = fallback_threshold
        self.timeout_threshold = timeout_threshold
        
        # Error tracking
        self.error_count = 0
        self.timeout_count = 0
        self.errors: List[ProcessingError] = []
        self.retry_counts: Dict[str, int] = {}
        
    def handle_error(self, error: Exception, task_id: str) -> ErrorAction:
        """Determine appropriate action for encountered error.
        
        Args:
            error: The exception that occurred
            task_id: Unique identifier for the task that failed
            
        Returns:
            ErrorAction indicating what to do next
        """
        current_time = time.time()
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track retry count for this specific task
        retry_count = self.retry_counts.get(task_id, 0)
        
        # Create error record
        processing_error = ProcessingError(
            task_id=task_id,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count,
            timestamp=current_time
        )
        self.errors.append(processing_error)
        
        # Handle timeout errors specially
        if "timeout" in error_message.lower() or error_type == "TimeoutError":
            self.timeout_count += 1
            logger.warning(f"Timeout error in task {task_id}: {error_message}")
            
            if self.timeout_count >= self.timeout_threshold:
                logger.warning(f"Too many timeouts ({self.timeout_count}), falling back to sequential processing")
                return ErrorAction.FALLBACK_TO_SEQUENTIAL
                
        # Handle worker process crashes
        if "process" in error_message.lower() and ("crash" in error_message.lower() or "died" in error_message.lower()):
            logger.error(f"Worker process crashed for task {task_id}: {error_message}")
            self.error_count += 1
            
            if self.error_count >= self.fallback_threshold:
                logger.error(f"Too many worker crashes ({self.error_count}), falling back to sequential processing")
                return ErrorAction.FALLBACK_TO_SEQUENTIAL
                
            return ErrorAction.RETRY
            
        # Handle general errors with retry logic
        if retry_count < self.max_retries:
            self.retry_counts[task_id] = retry_count + 1
            logger.debug(f"Retrying task {task_id} (attempt {retry_count + 1}/{self.max_retries}): {error_message}")
            return ErrorAction.RETRY
        else:
            logger.warning(f"Max retries exceeded for task {task_id}, skipping: {error_message}")
            self.error_count += 1
            
            if self.error_count >= self.fallback_threshold:
                logger.error(f"Too many errors ({self.error_count}), falling back to sequential processing")
                return ErrorAction.FALLBACK_TO_SEQUENTIAL
                
            return ErrorAction.SKIP
            
    def should_fallback_to_sequential(self) -> bool:
        """Check if we should abandon parallel processing."""
        return self.error_count >= self.fallback_threshold
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        error_types = {}
        for error in self.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
        return {
            "total_errors": len(self.errors),
            "unique_error_types": len(error_types),
            "error_breakdown": error_types,
            "timeout_count": self.timeout_count,
            "fallback_triggered": self.should_fallback_to_sequential()
        }
        
    def reset(self) -> None:
        """Reset error tracking state."""
        self.error_count = 0
        self.timeout_count = 0
        self.errors.clear()
        self.retry_counts.clear()
        
    def log_error_summary(self) -> None:
        """Log a summary of all errors encountered."""
        if not self.errors:
            return
            
        summary = self.get_error_summary()
        logger.info(f"Parallel processing error summary: {summary['total_errors']} total errors")
        
        for error_type, count in summary["error_breakdown"].items():
            logger.info(f"  {error_type}: {count} occurrences")
            
        if summary["fallback_triggered"]:
            logger.warning("Fallback to sequential processing was triggered due to excessive errors")