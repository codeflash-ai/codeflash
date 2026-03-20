from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JfrProfile:
    """Parses JFR (Java Flight Recorder) files for method-level profiling data.

    Uses the `jfr` CLI tool (ships with JDK 11+) to extract ExecutionSample events
    and build method-level timing estimates from sampling data.
    """

    def __init__(self, jfr_file: Path, packages: list[str]) -> None:
        self.jfr_file = jfr_file
        self.packages = packages
        self._method_samples: dict[str, int] = {}
        self._method_info: dict[str, dict[str, str]] = {}
        self._caller_map: dict[str, dict[str, int]] = {}
        self._recording_duration_ns: int = 0
        self._total_samples: int = 0
        self._parse()

    def _find_jfr_tool(self) -> str | None:
        jfr_path = shutil.which("jfr")
        if jfr_path:
            return jfr_path

        java_home = subprocess.run(
            ["java", "-XshowSettings:property", "-version"], capture_output=True, text=True, check=False
        )
        for line in java_home.stderr.splitlines():
            if "java.home" in line:
                home = line.split("=", 1)[1].strip()
                candidate = Path(home) / "bin" / "jfr"
                if candidate.exists():
                    return str(candidate)
        return None

    def _parse(self) -> None:
        if not self.jfr_file.exists():
            logger.warning("JFR file not found: %s", self.jfr_file)
            return

        jfr_tool = self._find_jfr_tool()
        if jfr_tool is None:
            logger.warning("jfr CLI tool not found, cannot parse JFR profile")
            return

        try:
            result = subprocess.run(
                [jfr_tool, "print", "--events", "jdk.ExecutionSample", "--json", str(self.jfr_file)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                logger.warning("jfr print failed: %s", result.stderr)
                return
            self._parse_json(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("jfr print timed out for %s", self.jfr_file)
        except Exception:
            logger.exception("Failed to parse JFR file %s", self.jfr_file)

    def _parse_json(self, json_str: str) -> None:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JFR JSON output")
            return

        events = data.get("recording", {}).get("events", [])
        if not events:
            events = data.get("events", [])

        # Cache package matching results to avoid repeated checks
        package_match_cache: dict[str, bool] = {}

        def matches_packages_cached(method_key: str | None) -> bool:
            if method_key is None:
                return False
            if method_key not in package_match_cache:
                package_match_cache[method_key] = self._matches_packages(method_key)
            return package_match_cache[method_key]

        for event in events:
            if event.get("type") != "jdk.ExecutionSample":
                continue

            stack_trace = event.get("values", {}).get("stackTrace", {})
            frames = stack_trace.get("frames", [])
            if not frames:
                continue

            self._total_samples += 1

            # Precompute keys for all frames in this stack to avoid repeated conversions
            keys = [self._frame_to_key(f) for f in frames]

            # Top-of-stack = own time
            top_method_key = keys[0] if keys else None
            if matches_packages_cached(top_method_key):
                self._method_samples[top_method_key] = self._method_samples.get(top_method_key, 0) + 1
                self._store_method_info(top_method_key, frames[0])

            # Build caller-callee relationships from adjacent frames
            for i in range(len(keys) - 1):
                callee_key = keys[i]
                caller_key = keys[i + 1]
                if callee_key and caller_key and matches_packages_cached(callee_key):
                    callee_callers = self._caller_map.setdefault(callee_key, {})
                    callee_callers[caller_key] = callee_callers.get(caller_key, 0) + 1

        # Estimate recording duration from event timestamps
        if events:
            min_ts = None
            max_ts = None
            for event in events:
                try:
                    start_time = event.get("values", {}).get("startTime")
                    if not start_time:
                        continue
                    # JFR timestamps are in ISO format or epoch nanos
                    if isinstance(start_time, str):
                        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        ts = int(dt.timestamp() * 1_000_000_000)
                    elif isinstance(start_time, (int, float)):
                        ts = int(start_time)
                    else:
                        continue
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts
                except (ValueError, TypeError):
                    continue
            if min_ts is not None and max_ts is not None:
                self._recording_duration_ns = max_ts - min_ts

    def _frame_to_key(self, frame: dict[str, Any]) -> str | None:
        method = frame.get("method", {})
        class_name = method.get("type", {}).get("name", "")
        method_name = method.get("name", "")
        if not class_name or not method_name:
            return None
        # JFR uses / separators (JVM internal format), normalize to dots for package matching
        class_name = class_name.replace("/", ".")
        return f"{class_name}.{method_name}"

    def _store_method_info(self, key: str, frame: dict[str, Any]) -> None:
        if key in self._method_info:
            return
        method = frame.get("method", {})
        self._method_info[key] = {
            "class_name": method.get("type", {}).get("name", "").replace("/", "."),
            "method_name": method.get("name", ""),
            "descriptor": method.get("descriptor", ""),
            "line_number": str(frame.get("lineNumber", 0)),
        }

    def _matches_packages(self, method_key: str) -> bool:
        if not self.packages:
            return True
        return any(method_key.startswith(pkg) for pkg in self.packages)

    def get_method_ranking(self) -> list[dict[str, Any]]:
        if not self._method_samples or self._total_samples == 0:
            return []

        ranking = []
        for method_key, sample_count in sorted(self._method_samples.items(), key=lambda x: x[1], reverse=True):
            info = self._method_info.get(method_key, {})
            ranking.append(
                {
                    "class_name": info.get("class_name", method_key.rsplit(".", 1)[0]),
                    "method_name": info.get("method_name", method_key.rsplit(".", 1)[-1]),
                    "sample_count": sample_count,
                    "pct_of_total": (sample_count / self._total_samples) * 100,
                }
            )
        return ranking

    def get_addressable_time_ns(self, class_name: str, method_name: str) -> float:
        method_key = f"{class_name}.{method_name}"
        sample_count = self._method_samples.get(method_key, 0)
        if sample_count == 0 or self._total_samples == 0:
            return 0.0

        if self._recording_duration_ns > 0:
            return (sample_count / self._total_samples) * self._recording_duration_ns

        # Fallback: return sample count as a proxy (higher = more time)
        return float(sample_count * 1_000_000)
