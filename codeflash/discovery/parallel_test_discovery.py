"""Main orchestrator for parallel test discovery."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from codeflash.cli_cmds.console import logger
from codeflash.discovery.parallel_models import (
    ImportAnalysisTask,
    JediAnalysisTask,
    ParallelConfig,
    ProgressInfo,
)
from codeflash.discovery.test_file_processor import TestFileProcessor
from codeflash.discovery.worker_pool import WorkerPool
from codeflash.models.models import FunctionCalledInTest, TestsInFile
from codeflash.verification.verification_utils import TestConfig


class ParallelTestDiscovery:
    """Main orchestrator for parallel test discovery."""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel test discovery with configuration.
        
        Args:
            config: Parallel processing configuration. If None, uses defaults.
        """
        self.config = config or ParallelConfig()
        self.progress_info: Optional[ProgressInfo] = None
        
    def discover_tests(
        self,
        cfg: TestConfig,
        file_to_test_map: Dict[Path, List[TestsInFile]],
        target_functions: Optional[Set[str]] = None
    ) -> Tuple[Dict[str, Set[FunctionCalledInTest]], int, int]:
        """Main entry point for parallel test discovery.
        
        Args:
            cfg: Test configuration
            file_to_test_map: Mapping of test files to test functions
            target_functions: Set of function names to optimize (for import filtering)
            
        Returns:
            Tuple of (function_to_test_mapping, num_discovered_tests, num_discovered_replay_tests)
        """
        if not file_to_test_map:
            return {}, 0, 0
            
        start_time = time.time()
        
        # Step 1: Filter test files by imports if target functions are provided
        if target_functions:
            logger.debug(f"Filtering {len(file_to_test_map)} test files by imports for {len(target_functions)} target functions")
            file_to_test_map = self._filter_files_by_imports(file_to_test_map, target_functions)
            logger.debug(f"After import filtering: {len(file_to_test_map)} test files remain")
            
        if not file_to_test_map:
            return {}, 0, 0
            
        # Step 2: Process remaining files with Jedi analysis
        logger.debug(f"Processing {len(file_to_test_map)} test files with Jedi analysis")
        function_to_test_map, num_discovered_tests, num_discovered_replay_tests = self._process_files_with_jedi(
            file_to_test_map, cfg
        )
        
        total_time = time.time() - start_time
        logger.info(f"Parallel test discovery completed in {total_time:.2f}s: {num_discovered_tests} tests, {num_discovered_replay_tests} replay tests")
        
        return function_to_test_map, num_discovered_tests, num_discovered_replay_tests
        
    def _filter_files_by_imports(
        self,
        file_to_test_map: Dict[Path, List[TestsInFile]],
        target_functions: Set[str]
    ) -> Dict[Path, List[TestsInFile]]:
        """Filter test files based on import analysis to reduce Jedi processing."""
        if not self.config.enable_parallel:
            return self._filter_files_by_imports_sequential(file_to_test_map, target_functions)
            
        # Create import analysis tasks
        tasks = [
            ImportAnalysisTask(test_file_path=test_file, target_functions=target_functions)
            for test_file in file_to_test_map.keys()
        ]
        
        # Initialize progress tracking
        self.progress_info = ProgressInfo(
            total_files=len(tasks),
            completed_files=0,
            failed_files=0,
            current_phase="import_analysis",
            start_time=time.time()
        )
        
        # Process tasks in parallel
        with WorkerPool(self.config) as pool:
            try:
                results = pool.map_parallel(
                    TestFileProcessor.process_import_analysis,
                    tasks,
                    timeout=self.config.timeout_seconds
                )
            except Exception as e:
                logger.warning(f"Parallel import analysis failed: {e}, falling back to sequential")
                return self._filter_files_by_imports_sequential(file_to_test_map, target_functions)
                
        # Process results
        filtered_map = {}
        failed_count = 0
        
        for result in results:
            if result is None:
                failed_count += 1
                continue
                
            if result.error:
                logger.debug(f"Import analysis error for {result.test_file_path}: {result.error}")
                failed_count += 1
                # Include file on error to be safe
                if result.test_file_path in file_to_test_map:
                    filtered_map[result.test_file_path] = file_to_test_map[result.test_file_path]
            elif result.has_target_imports:
                if result.test_file_path in file_to_test_map:
                    filtered_map[result.test_file_path] = file_to_test_map[result.test_file_path]
                    
        logger.debug(f"Import analysis completed: {len(filtered_map)} files passed, {failed_count} failed")
        return filtered_map
        
    def _filter_files_by_imports_sequential(
        self,
        file_to_test_map: Dict[Path, List[TestsInFile]],
        target_functions: Set[str]
    ) -> Dict[Path, List[TestsInFile]]:
        """Sequential fallback for import filtering."""
        filtered_map = {}
        
        for test_file in file_to_test_map.keys():
            task = ImportAnalysisTask(test_file_path=test_file, target_functions=target_functions)
            result = TestFileProcessor.process_import_analysis(task)
            
            if result.has_target_imports:
                filtered_map[test_file] = file_to_test_map[test_file]
                
        return filtered_map
        
    def _process_files_with_jedi(
        self,
        file_to_test_map: Dict[Path, List[TestsInFile]],
        cfg: TestConfig
    ) -> Tuple[Dict[str, Set[FunctionCalledInTest]], int, int]:
        """Process test files with Jedi analysis to find function references."""
        if not self.config.enable_parallel:
            return self._process_files_with_jedi_sequential(file_to_test_map, cfg)
            
        # Create Jedi analysis tasks
        tasks = [
            JediAnalysisTask(
                test_file=test_file,
                test_functions=test_functions,
                project_root=cfg.project_root_path,
                test_framework=cfg.test_framework
            )
            for test_file, test_functions in file_to_test_map.items()
        ]
        
        # Update progress tracking
        if self.progress_info:
            self.progress_info.current_phase = "jedi_processing"
            self.progress_info.total_files = len(tasks)
            self.progress_info.completed_files = 0
            self.progress_info.failed_files = 0
        else:
            self.progress_info = ProgressInfo(
                total_files=len(tasks),
                completed_files=0,
                failed_files=0,
                current_phase="jedi_processing",
                start_time=time.time()
            )
            
        # Process tasks in parallel
        with WorkerPool(self.config) as pool:
            try:
                results = pool.map_parallel(
                    TestFileProcessor.process_jedi_analysis,
                    tasks,
                    timeout=self.config.timeout_seconds
                )
            except Exception as e:
                logger.warning(f"Parallel Jedi analysis failed: {e}, falling back to sequential")
                return self._process_files_with_jedi_sequential(file_to_test_map, cfg)
                
        # Aggregate results
        function_to_test_map = defaultdict(set)
        total_discovered_tests = 0
        total_replay_tests = 0
        failed_count = 0
        
        for result in results:
            if result is None:
                failed_count += 1
                continue
                
            if result.error:
                logger.debug(f"Jedi analysis error for {result.test_file}: {result.error}")
                failed_count += 1
                continue
                
            # Merge function mappings
            for func_name, test_set in result.function_mappings.items():
                function_to_test_map[func_name].update(test_set)
                
            total_discovered_tests += result.test_count
            total_replay_tests += result.replay_test_count
            
        logger.debug(f"Jedi analysis completed: {len(results) - failed_count} files processed, {failed_count} failed")
        return dict(function_to_test_map), total_discovered_tests, total_replay_tests
        
    def _process_files_with_jedi_sequential(
        self,
        file_to_test_map: Dict[Path, List[TestsInFile]],
        cfg: TestConfig
    ) -> Tuple[Dict[str, Set[FunctionCalledInTest]], int, int]:
        """Sequential fallback for Jedi processing."""
        function_to_test_map = defaultdict(set)
        total_discovered_tests = 0
        total_replay_tests = 0
        
        for test_file, test_functions in file_to_test_map.items():
            task = JediAnalysisTask(
                test_file=test_file,
                test_functions=test_functions,
                project_root=cfg.project_root_path,
                test_framework=cfg.test_framework
            )
            
            result = TestFileProcessor.process_jedi_analysis(task)
            
            if not result.error:
                # Merge function mappings
                for func_name, test_set in result.function_mappings.items():
                    function_to_test_map[func_name].update(test_set)
                    
                total_discovered_tests += result.test_count
                total_replay_tests += result.replay_test_count
                
        return dict(function_to_test_map), total_discovered_tests, total_replay_tests
        
    def get_progress_info(self) -> Optional[ProgressInfo]:
        """Get current progress information."""
        return self.progress_info