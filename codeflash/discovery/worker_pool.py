"""Worker pool implementation for parallel test discovery processing."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from typing import Any, Callable, List, Optional

from codeflash.cli_cmds.console import logger
from codeflash.discovery.parallel_error_handler import ParallelErrorHandler
from codeflash.discovery.parallel_models import ParallelConfig


class WorkerPool:
    """Manages worker processes and task distribution for parallel processing."""
    
    def __init__(self, config: ParallelConfig):
        """Initialize worker pool with configuration.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self.max_workers = self._determine_worker_count()
        self.executor: Optional[ProcessPoolExecutor] = None
        self.error_handler = ParallelErrorHandler()
        
    def _determine_worker_count(self) -> int:
        """Determine optimal number of workers based on configuration and system resources."""
        if self.config.max_workers is not None:
            return max(1, self.config.max_workers)
            
        # Auto-detect based on CPU cores, but cap at reasonable limits
        cpu_count = os.cpu_count() or 1
        
        # Use 75% of available cores, but at least 1 and at most 8 for test discovery
        optimal_workers = max(1, min(8, int(cpu_count * 0.75)))
        
        logger.debug(f"Auto-detected {optimal_workers} workers for parallel processing (CPU cores: {cpu_count})")
        return optimal_workers
        
    def map_parallel(
        self,
        func: Callable[[Any], Any],
        tasks: List[Any],
        timeout: Optional[int] = None
    ) -> List[Any]:
        """Execute function on tasks in parallel across workers.
        
        Args:
            func: Function to execute on each task
            tasks: List of tasks to process
            timeout: Optional timeout per task in seconds
            
        Returns:
            List of results in the same order as input tasks
            
        Raises:
            Exception: If parallel processing fails and fallback is disabled
        """
        if not tasks:
            return []
            
        if not self.config.enable_parallel or self.max_workers == 1:
            logger.debug("Running tasks sequentially (parallel processing disabled or single worker)")
            return [func(task) for task in tasks]
            
        timeout = timeout or self.config.timeout_seconds
        results = [None] * len(tasks)
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                self.executor = executor
                
                # Submit all tasks
                future_to_index = {}
                for i, task in enumerate(tasks):
                    future = executor.submit(func, task)
                    future_to_index[future] = i
                    
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_index.keys(), timeout=timeout * len(tasks)):
                    task_index = future_to_index[future]
                    task_id = f"task_{task_index}"
                    
                    try:
                        result = future.result(timeout=timeout)
                        results[task_index] = result
                        completed_count += 1
                        
                    except TimeoutError as e:
                        error_action = self.error_handler.handle_error(e, task_id)
                        if error_action.value == "fallback_to_sequential":
                            logger.warning("Falling back to sequential processing due to timeouts")
                            return self._fallback_to_sequential(func, tasks)
                        else:
                            # For timeout, we'll skip this task and continue
                            logger.warning(f"Task {task_index} timed out, skipping")
                            completed_count += 1
                            
                    except Exception as e:
                        error_action = self.error_handler.handle_error(e, task_id)
                        if error_action.value == "fallback_to_sequential":
                            logger.warning("Falling back to sequential processing due to errors")
                            return self._fallback_to_sequential(func, tasks)
                        else:
                            # For other errors, we'll skip this task and continue
                            logger.warning(f"Task {task_index} failed: {e}")
                            completed_count += 1
                            
                logger.debug(f"Parallel processing completed: {completed_count}/{len(tasks)} tasks")
                
        except Exception as e:
            logger.error(f"Worker pool execution failed: {e}")
            if self.config.fallback_on_error:
                logger.info("Falling back to sequential processing")
                return self._fallback_to_sequential(func, tasks)
            else:
                raise
                
        finally:
            self.executor = None
            
        # Filter out None results (failed tasks)
        return [result for result in results if result is not None]
        
    def _fallback_to_sequential(self, func: Callable[[Any], Any], tasks: List[Any]) -> List[Any]:
        """Fallback to sequential processing when parallel processing fails."""
        logger.info(f"Processing {len(tasks)} tasks sequentially as fallback")
        results = []
        
        for i, task in enumerate(tasks):
            try:
                result = func(task)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sequential task {i} failed: {e}")
                # Continue with other tasks even if one fails
                continue
                
        return results
        
    def shutdown(self) -> None:
        """Clean shutdown of all worker processes."""
        if self.executor:
            try:
                # Try graceful shutdown first
                self.executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.warning(f"Error during executor shutdown: {e}")
                
        self.executor = None
        
    def get_worker_count(self) -> int:
        """Get the current number of workers."""
        return self.max_workers
        
    def adjust_worker_count(self, new_count: int) -> None:
        """Dynamically adjust worker count (requires restart of executor)."""
        if new_count != self.max_workers:
            logger.info(f"Adjusting worker count from {self.max_workers} to {new_count}")
            self.max_workers = max(1, new_count)
            
            # If executor is running, we'll need to restart it
            if self.executor:
                self.shutdown()
                
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()