"""Benchmark CLI startup latency for codeflash compare --script mode.

Run from a worktree root. Installs deps via uv sync, then times several
CLI entry points and writes a JSON file mapping command names to median
wall-clock seconds.

Usage:
    codeflash compare main codeflash/optimize \
        --script "python benchmarks/bench_cli_startup.py" \
        --script-output benchmarks/results.json
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

WARMUP = 3
RUNS = 30
OUTPUT = os.environ.get("BENCH_OUTPUT", "benchmarks/results.json")

COMMANDS: dict[str, list[str]] = {
    "version": ["uv", "run", "codeflash", "--version"],
    "help": ["uv", "run", "codeflash", "--help"],
    "auth_status": ["uv", "run", "codeflash", "auth", "status"],
    "compare_help": ["uv", "run", "codeflash", "compare", "--help"],
}


def measure(cmd: list[str], warmup: int = WARMUP, runs: int = RUNS) -> float:
    """Return median wall-clock seconds for *cmd* over *runs* iterations."""
    env = {**os.environ, "CODEFLASH_API_KEY": "bench_dummy_key"}
    for _ in range(warmup):
        subprocess.run(cmd, capture_output=True, check=False, env=env)

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, check=False, env=env)
        times.append(time.perf_counter() - t0)

    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2


def main() -> None:
    # Ensure deps are installed in the worktree
    subprocess.run(["uv", "sync"], check=True, capture_output=True)

    results: dict[str, float] = {}
    for name, cmd in COMMANDS.items():
        print(f"  {name}: ", end="", flush=True)
        median = measure(cmd)
        results[name] = round(median, 4)
        print(f"{median * 1000:.0f} ms")

    # Total = sum of medians (useful for a single summary number)
    results["__total__"] = round(sum(results.values()), 4)

    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT}")


if __name__ == "__main__":
    main()
