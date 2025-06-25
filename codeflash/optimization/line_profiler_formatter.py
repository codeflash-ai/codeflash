import json
import re
import platform
import os
import socket
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging


@dataclass
class ProfileLine:
    line_number: int
    hits: int
    time: float
    time_per_hit: float
    percent_time: float
    code: str


@dataclass
class ProfileFunction:
    function_name: str
    file: str
    start_line: int
    total_time: float
    lines: List[ProfileLine] = field(default_factory=list)


class SystemInfoCollector:
    """Collects basic system information using only standard library."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def collect(self) -> Dict[str, Any]:
        return {
            "system": self._collect_system_info(),
            "environment": self._collect_environment_info(),
        }

    def _collect_system_info(self) -> Dict[str, Any]:
        info = {
            "platform": platform.platform(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": socket.gethostname(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_version_info": list(sys.version_info),
            "python_implementation": platform.python_implementation(),
            "cpu_count": os.cpu_count(),
        }

        # OS-specific info
        if platform.system() == "Darwin":
            info["mac_version"] = platform.mac_ver()[0]
        elif platform.system() == "Linux":
            try:
                import distro
                info["linux_distribution"] = {
                    "name": distro.name(),
                    "version": distro.version(),
                    "codename": distro.codename(),
                }
            except ImportError:
                pass
        elif platform.system() == "Windows":
            info["windows_version"] = platform.win32_ver()[0]

        return info

    def _collect_environment_info(self) -> Dict[str, str]:
        return {
            "python_path": os.environ.get("PYTHONPATH", ""),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "")),
            "shell": os.environ.get("SHELL", ""),
            "term": os.environ.get("TERM", ""),
        }


class ProfilerParser:
    """Parses line profiler output into structured data."""

    def parse(self, output: str) -> List[ProfileFunction]:
        functions = []
        current_func = None
        lines = output.strip().split('\n')

        # Track if we're in a table and line numbering
        in_table = False
        actual_line_number = 0
        function_start_line = 0

        for i, line in enumerate(lines):
            # Check for function header (new format)
            if line.startswith("## Function:"):
                if current_func:
                    functions.append(current_func)

                func_name = line.replace("## Function:", "").strip()
                current_func = ProfileFunction(
                    function_name=func_name,
                    file="",  # Will be extracted from line contents or context
                    start_line=0,
                    total_time=0.0
                )
                in_table = False

            # Check for total time (new format)
            elif line.startswith("## Total time:") and current_func:
                time_match = re.search(r'Total time: ([\d.]+) s', line)
                if time_match:
                    current_func.total_time = float(time_match.group(1))

            # Old format: Total time
            elif "Total time:" in line and not line.startswith("##"):
                if current_func:
                    functions.append(current_func)
                current_func = self._create_function_from_header(lines, i)
                in_table = False

            # Check if we're entering a table
            elif "|" in line and "Hits" in line and "Time" in line:
                in_table = True
                # Next content lines will be actual code
                actual_line_number = 0
                continue

            # Skip separator lines
            elif line.strip().startswith("|-"):
                continue

            # Parse table rows (new format)
            elif in_table and "|" in line and current_func:
                profile_line = self._parse_table_line(line)
                if profile_line and not self._should_skip_line(profile_line):
                    # Determine actual line number based on code content
                    if "def " in profile_line.code and function_start_line == 0:
                        # This is likely the function definition
                        function_start_line = self._estimate_start_line(profile_line.code)
                        if function_start_line == 0:
                            function_start_line = 1  # Default
                        current_func.start_line = function_start_line
                        actual_line_number = function_start_line
                    else:
                        actual_line_number += 1

                    # Update line number
                    profile_line.line_number = actual_line_number

                    # Extract file path from first code line if needed
                    if not current_func.file:
                        current_func.file = self._extract_file_from_context(lines, i)

                    current_func.lines.append(profile_line)

            # Old format line parsing
            elif current_func and not in_table:
                profile_line = self._parse_profile_line(line)
                if profile_line and not self._should_skip_line(profile_line):
                    current_func.lines.append(profile_line)

        if current_func:
            functions.append(current_func)

        return functions

    def _should_skip_line(self, profile_line: ProfileLine) -> bool:
        """Check if a line should be skipped - only skip codeflash decorator."""
        # Only skip the codeflash_line_profile decorator
        return "@codeflash_line_profile" in profile_line.code

    def _parse_table_line(self, line: str) -> Optional[ProfileLine]:
        """Parse a markdown table line from line profiler output."""
        # Split by | and strip whitespace
        parts = [p.strip() for p in line.split('|')]

        # Filter out empty parts at beginning and end
        if parts and parts[0] == '':
            parts = parts[1:]
        if parts and parts[-1] == '':
            parts = parts[:-1]

        if len(parts) >= 5:
            try:
                # Parse hits
                hits_str = parts[0].strip()
                if not hits_str:
                    hits = 0
                else:
                    hits = int(hits_str)

                # Parse time (handle scientific notation)
                time_str = parts[1].strip()
                if not time_str:
                    time = 0.0
                else:
                    # Handle scientific notation like 1e+10
                    time = float(time_str.replace(' ', ''))

                # Parse per hit
                per_hit_str = parts[2].strip()
                if not per_hit_str:
                    per_hit = 0.0
                else:
                    per_hit = float(per_hit_str)

                # Parse percent
                percent_str = parts[3].strip()
                if not percent_str:
                    percent = 0.0
                else:
                    percent = float(percent_str)

                # Get code (handle case where code might contain |)
                code = parts[4]
                if len(parts) > 5:
                    # Join remaining parts in case code contained |
                    code = ' | '.join(parts[4:])

                return ProfileLine(
                    line_number=0,  # Will be set later
                    hits=hits,
                    time=time,
                    time_per_hit=per_hit,
                    percent_time=percent,
                    code=code
                )
            except (ValueError, IndexError) as e:
                # This might be a code-only line or parsing error
                # Try to parse as code-only line
                if len(parts) >= 5 and not parts[0] and not parts[1]:
                    # Empty hits and time, just code
                    code = parts[4]
                    if len(parts) > 5:
                        code = ' | '.join(parts[4:])

                    return ProfileLine(
                        line_number=0,  # Will be set later
                        hits=0,
                        time=0.0,
                        time_per_hit=0.0,
                        percent_time=0.0,
                        code=code
                    )

        return None

    def _estimate_start_line(self, code_line: str) -> int:
        """Try to estimate the start line number from the function definition."""
        # This is a placeholder - in real scenario, you might have this info elsewhere
        # For now, we'll use a default
        return 1

    def _extract_file_from_context(self, lines: List[str], current_index: int) -> str:
        """Try to extract file path from context."""
        # Look backwards for file information
        for i in range(current_index - 1, max(0, current_index - 10), -1):
            if "File:" in lines[i]:
                file_match = re.search(r'File: (.+)', lines[i])
                if file_match:
                    return file_match.group(1).strip()

        # Look for file in comments or other markers
        for i in range(max(0, current_index - 10), min(len(lines), current_index + 5)):
            if "# File:" in lines[i] or "## File:" in lines[i]:
                file_match = re.search(r'File:\s*(.+)', lines[i])
                if file_match:
                    return file_match.group(1).strip()

        # If no file found, return empty
        return ""

    def _create_function_from_header(self, lines: List[str], start_index: int) -> ProfileFunction:
        """Parse old format headers."""
        time_match = re.search(r'Total time: ([\d.]+) s', lines[start_index])
        total_time = float(time_match.group(1)) if time_match else 0.0

        func = ProfileFunction(
            function_name="",
            file="",
            start_line=0,
            total_time=total_time
        )

        for j in range(start_index + 1, min(start_index + 5, len(lines))):
            if "File:" in lines[j]:
                file_match = re.search(r'File: (.+)', lines[j])
                if file_match:
                    func.file = file_match.group(1).strip()

            if "Function:" in lines[j]:
                func_match = re.search(r'Function: (\w+) at line (\d+)', lines[j])
                if func_match:
                    func.function_name = func_match.group(1)
                    func.start_line = int(func_match.group(2))

        return func

    def _parse_profile_line(self, line: str) -> Optional[ProfileLine]:
        """Parse old format lines."""
        match = re.match(
            r'\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%?\s+(.+)',
            line
        )

        if match:
            return ProfileLine(
                line_number=int(match.group(1)),
                hits=int(match.group(2)),
                time=float(match.group(3)),
                time_per_hit=float(match.group(4)),
                percent_time=float(match.group(5)),
                code=match.group(6).strip()
            )
        return None


class ProfilerFormatter:
    """Formats profiler results for storage and display."""

    def __init__(
        self,
        parser: ProfilerParser,
        system_collector: Optional[SystemInfoCollector] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.parser = parser
        self.system_collector = system_collector
        self.logger = logger or logging.getLogger(__name__)

    def format(
        self,
        line_profile_results: Dict[str, Any],
        test_name: Optional[str] = None,
        test_file: Optional[str] = None,
        include_machine_details: bool = True
    ) -> Dict[str, Any]:
        str_output = line_profile_results.get("str_out", "")

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_name": test_name,
            "test_file": test_file,
            "summary": {},
            "functions": [],
            "raw_output": str_output
        }

        if include_machine_details and self.system_collector:
            result["machine_info"] = self.system_collector.collect()
            self._log_machine_summary(result["machine_info"])

        functions = self.parser.parse(str_output)
        result["functions"] = [self._function_to_dict(f) for f in functions]

        if functions:
            result["summary"] = self._calculate_summary(functions, result.get("machine_info"))

        self._log_summary(result)
        return result

    def _function_to_dict(self, func: ProfileFunction) -> Dict[str, Any]:
        return {
            "function_name": func.function_name,
            "file": func.file,
            "start_line": func.start_line,
            "total_time": func.total_time,
            "lines": [asdict(line) for line in func.lines]
        }

    def _calculate_summary(
        self,
        functions: List[ProfileFunction],
        machine_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_time = sum(f.total_time for f in functions)
        total_lines = sum(len(f.lines) for f in functions)

        summary = {
            "total_time_seconds": total_time,
            "function_count": len(functions),
            "total_lines_profiled": total_lines,
            "average_time_per_function": total_time / len(functions) if functions else 0
        }

        if machine_info and machine_info.get("system", {}).get("cpu_count"):
            cpu_cores = machine_info["system"]["cpu_count"]
            summary["normalized_time"] = total_time / cpu_cores
            summary["time_per_core"] = total_time / cpu_cores

        return summary

    def _log_machine_summary(self, machine_info: Dict[str, Any]):
        self.logger.info("Machine details:")
        self.logger.info(f"  Platform: {machine_info['system']['platform']}")
        self.logger.info(f"  Python: {machine_info['system']['python_version'].split()[0]}")
        self.logger.info(f"  CPU cores: {machine_info['system'].get('cpu_count', 'N/A')}")
        self.logger.info(f"  Hostname: {machine_info['system']['hostname']}")

    def _log_summary(self, result: Dict[str, Any]):
        self.logger.info("Line profiler summary:")
        self.logger.info(f"  Total time: {result['summary'].get('total_time_seconds', 0):.3f}s")
        self.logger.info(f"  Functions: {result['summary'].get('function_count', 0)}")
        self.logger.info(f"  Lines profiled: {result['summary'].get('total_lines_profiled', 0)}")


def format_line_profiler_results(
    line_profile_results: Dict[str, Any],
    test_name: Optional[str] = None,
    test_file: Optional[str] = None,
    include_machine_details: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Format line profiler results.

    Args:
        line_profile_results: Dict with 'str_out' key containing profiler output
        test_name: Optional test name for context
        test_file: Optional test file path
        include_machine_details: Whether to include machine/system details
        logger: Optional logger instance

    Returns:
        Dict formatted for JSON storage with parsed profiler data
    """
    parser = ProfilerParser()
    system_collector = SystemInfoCollector(logger=logger) if include_machine_details else None
    formatter = ProfilerFormatter(parser, system_collector, logger)

    return formatter.format(
        line_profile_results,
        test_name=test_name,
        test_file=test_file,
        include_machine_details=include_machine_details
    )
