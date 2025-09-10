"""Tests for parallel test discovery infrastructure."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codeflash.discovery.parallel_error_handler import ErrorAction, ParallelErrorHandler
from codeflash.discovery.parallel_models import (
    ImportAnalysisResult,
    ImportAnalysisTask,
    ParallelConfig,
    ProgressInfo,
)
from codeflash.discovery.test_file_processor import TestFileProcessor
from codeflash.discovery.worker_pool import WorkerPool


class TestParallelConfig:
    """Test ParallelConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelConfig()
        assert config.max_workers is None
        assert config.enable_parallel is True
        assert config.chunk_size == 1
        assert config.timeout_seconds == 300
        assert config.fallback_on_error is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParallelConfig(
            max_workers=4,
            enable_parallel=False,
            chunk_size=2,
            timeout_seconds=60,
            fallback_on_error=False
        )
        assert config.max_workers == 4
        assert config.enable_parallel is False
        assert config.chunk_size == 2
        assert config.timeout_seconds == 60
        assert config.fallback_on_error is False


class TestProgressInfo:
    """Test ProgressInfo data class."""
    
    def test_progress_info_creation(self):
        """Test creating progress info."""
        start_time = time.time()
        progress = ProgressInfo(
            total_files=10,
            completed_files=5,
            failed_files=1,
            current_phase="import_analysis",
            start_time=start_time
        )
        
        assert progress.total_files == 10
        assert progress.completed_files == 5
        assert progress.failed_files == 1
        assert progress.current_phase == "import_analysis"
        assert progress.start_time == start_time
        assert progress.estimated_remaining is None


class TestParallelErrorHandler:
    """Test ParallelErrorHandler class."""
    
    def test_initialization(self):
        """Test error handler initialization."""
        handler = ParallelErrorHandler(max_retries=2, fallback_threshold=3)
        assert handler.max_retries == 2
        assert handler.fallback_threshold == 3
        assert handler.error_count == 0
        assert len(handler.errors) == 0
        
    def test_handle_retry_logic(self):
        """Test retry logic for errors."""
        handler = ParallelErrorHandler(max_retries=2)
        
        # First error should trigger retry
        action = handler.handle_error(ValueError("test error"), "task_1")
        assert action == ErrorAction.RETRY
        assert handler.retry_counts["task_1"] == 1
        
        # Second error should trigger retry
        action = handler.handle_error(ValueError("test error"), "task_1")
        assert action == ErrorAction.RETRY
        assert handler.retry_counts["task_1"] == 2
        
        # Third error should skip (max retries exceeded)
        action = handler.handle_error(ValueError("test error"), "task_1")
        assert action == ErrorAction.SKIP
        assert handler.error_count == 1
        
    def test_fallback_threshold(self):
        """Test fallback to sequential when error threshold is reached."""
        handler = ParallelErrorHandler(max_retries=0, fallback_threshold=2)
        
        # First error
        action = handler.handle_error(ValueError("error 1"), "task_1")
        assert action == ErrorAction.SKIP
        assert not handler.should_fallback_to_sequential()
        
        # Second error should trigger fallback
        action = handler.handle_error(ValueError("error 2"), "task_2")
        assert action == ErrorAction.FALLBACK_TO_SEQUENTIAL
        assert handler.should_fallback_to_sequential()
        
    def test_timeout_handling(self):
        """Test special handling of timeout errors."""
        handler = ParallelErrorHandler(timeout_threshold=2)
        
        # First timeout
        action = handler.handle_error(TimeoutError("timeout"), "task_1")
        assert handler.timeout_count == 1
        
        # Second timeout should trigger fallback
        action = handler.handle_error(TimeoutError("timeout"), "task_2")
        assert action == ErrorAction.FALLBACK_TO_SEQUENTIAL
        assert handler.timeout_count == 2
        
    def test_error_summary(self):
        """Test error summary generation."""
        handler = ParallelErrorHandler()
        
        handler.handle_error(ValueError("value error"), "task_1")
        handler.handle_error(TypeError("type error"), "task_2")
        handler.handle_error(ValueError("another value error"), "task_3")
        
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 3
        assert summary["unique_error_types"] == 2
        assert summary["error_breakdown"]["ValueError"] == 2
        assert summary["error_breakdown"]["TypeError"] == 1


class TestWorkerPool:
    """Test WorkerPool class."""
    
    def test_worker_count_determination(self):
        """Test automatic worker count determination."""
        config = ParallelConfig(max_workers=None)
        pool = WorkerPool(config)
        
        # Should auto-detect based on CPU cores
        assert pool.max_workers >= 1
        assert pool.max_workers <= 8  # Capped at 8
        
    def test_explicit_worker_count(self):
        """Test explicit worker count setting."""
        config = ParallelConfig(max_workers=4)
        pool = WorkerPool(config)
        assert pool.max_workers == 4
        
    def test_sequential_processing(self):
        """Test sequential processing when parallel is disabled."""
        config = ParallelConfig(enable_parallel=False)
        pool = WorkerPool(config)
        
        def simple_task(x):
            return x * 2
            
        tasks = [1, 2, 3, 4, 5]
        results = pool.map_parallel(simple_task, tasks)
        
        assert results == [2, 4, 6, 8, 10]
        
    def test_empty_task_list(self):
        """Test handling of empty task list."""
        config = ParallelConfig()
        pool = WorkerPool(config)
        
        results = pool.map_parallel(lambda x: x, [])
        assert results == []


class TestTestFileProcessor:
    """Test TestFileProcessor class."""
    
    def test_import_analysis_with_target_function(self):
        """Test import analysis when target function is imported."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
from pathlib import Path

def test_something():
    path = Path("test")
    return os.path.exists(str(path))
""")
            temp_file = Path(f.name)
            
        try:
            task = ImportAnalysisTask(
                test_file_path=temp_file,
                target_functions={"os.path.exists", "Path"}
            )
            
            result = TestFileProcessor.process_import_analysis(task)
            
            assert result.test_file_path == temp_file
            assert result.has_target_imports is True
            assert result.error is None
            
        finally:
            temp_file.unlink()
            
    def test_import_analysis_without_target_function(self):
        """Test import analysis when target function is not imported."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import json

def test_something():
    data = {"key": "value"}
    return json.dumps(data)
""")
            temp_file = Path(f.name)
            
        try:
            task = ImportAnalysisTask(
                test_file_path=temp_file,
                target_functions={"os.path.exists", "Path"}
            )
            
            result = TestFileProcessor.process_import_analysis(task)
            
            assert result.test_file_path == temp_file
            assert result.has_target_imports is False
            assert result.error is None
            
        finally:
            temp_file.unlink()
            
    def test_import_analysis_syntax_error(self):
        """Test import analysis with syntax error in file."""
        # Create a temporary test file with syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
def test_something(
    # Missing closing parenthesis
""")
            temp_file = Path(f.name)
            
        try:
            task = ImportAnalysisTask(
                test_file_path=temp_file,
                target_functions={"os.path.exists"}
            )
            
            result = TestFileProcessor.process_import_analysis(task)
            
            assert result.test_file_path == temp_file
            assert result.has_target_imports is True  # Should return True on error to be safe
            assert result.error is not None
            
        finally:
            temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])