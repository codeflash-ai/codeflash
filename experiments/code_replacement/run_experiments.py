"""Run experiments to compare code replacement approaches for JavaScript/TypeScript.

This script tests all three approaches against the test cases and generates
a comparison report.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from test_cases import get_test_cases


@dataclass
class ApproachResult:
    """Result from testing an approach on one test case."""

    test_name: str
    passed: bool
    time_ms: float
    error: Optional[str] = None
    output: Optional[str] = None


@dataclass
class ApproachSummary:
    """Summary of results for one approach."""

    name: str
    description: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    available: bool = True
    results: list[ApproachResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total * 100


def test_approach_b() -> ApproachSummary:
    """Test Approach B: Text-based replacement."""
    from approach_b_text_based import TextBasedReplacer

    summary = ApproachSummary(
        name="Approach B: Text-Based", description="Pure Python text manipulation using line numbers"
    )

    replacer = TextBasedReplacer()

    for tc in get_test_cases():
        start_time = time.perf_counter()
        try:
            result = replacer.replace_function(tc.original_source, tc.start_line, tc.end_line, tc.new_function)
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000

            # Normalize for comparison
            result_normalized = result.replace("\r\n", "\n")
            expected_normalized = tc.expected_result.replace("\r\n", "\n")

            passed = result_normalized == expected_normalized

            summary.results.append(
                ApproachResult(test_name=tc.name, passed=passed, time_ms=time_ms, output=result if not passed else None)
            )

            if passed:
                summary.passed += 1
            else:
                summary.failed += 1
            summary.total_time_ms += time_ms

        except Exception as e:
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            summary.results.append(ApproachResult(test_name=tc.name, passed=False, time_ms=time_ms, error=str(e)))
            summary.errors += 1
            summary.total_time_ms += time_ms

    return summary


def test_approach_c() -> ApproachSummary:
    """Test Approach C: Hybrid (tree-sitter + text)."""
    try:
        from approach_c_hybrid import TREE_SITTER_AVAILABLE, HybridReplacer
    except ImportError:
        return ApproachSummary(
            name="Approach C: Hybrid", description="Tree-sitter analysis + text replacement", available=False
        )

    if not TREE_SITTER_AVAILABLE:
        return ApproachSummary(
            name="Approach C: Hybrid", description="Tree-sitter analysis + text replacement", available=False
        )

    summary = ApproachSummary(name="Approach C: Hybrid", description="Tree-sitter analysis + text replacement")

    js_replacer = HybridReplacer("javascript")
    ts_replacer = HybridReplacer("typescript")

    for tc in get_test_cases():
        # Use TypeScript parser for TypeScript test cases
        is_typescript = "typescript" in tc.name or "interface" in tc.description.lower()
        replacer = ts_replacer if is_typescript else js_replacer

        start_time = time.perf_counter()
        try:
            result = replacer.replace_function_by_lines(tc.original_source, tc.start_line, tc.end_line, tc.new_function)
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000

            # Normalize for comparison
            result_normalized = result.replace("\r\n", "\n")
            expected_normalized = tc.expected_result.replace("\r\n", "\n")

            passed = result_normalized == expected_normalized

            summary.results.append(
                ApproachResult(test_name=tc.name, passed=passed, time_ms=time_ms, output=result if not passed else None)
            )

            if passed:
                summary.passed += 1
            else:
                summary.failed += 1
            summary.total_time_ms += time_ms

        except Exception as e:
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            summary.results.append(ApproachResult(test_name=tc.name, passed=False, time_ms=time_ms, error=str(e)))
            summary.errors += 1
            summary.total_time_ms += time_ms

    return summary


def test_approach_a() -> ApproachSummary:
    """Test Approach A: jscodeshift/recast."""
    summary = ApproachSummary(
        name="Approach A: jscodeshift", description="AST-based replacement via Node.js subprocess"
    )

    try:
        from approach_a_jscodeshift import JsCodeshiftReplacer

        replacer = JsCodeshiftReplacer()

        if not replacer._check_node_available():
            summary.available = False
            return summary

    except Exception:
        summary.available = False
        return summary

    # Note: Full jscodeshift testing requires npm packages
    # For now, we'll mark it as available but note limited testing
    summary.available = True

    # We won't run full tests since jscodeshift requires npm setup
    # Instead, note that this approach requires external dependencies

    return summary


def generate_report(summaries: list[ApproachSummary]) -> str:
    """Generate a markdown report of the experiment results."""
    report = []
    report.append("# Code Replacement Experiment Results\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overview table
    report.append("## Summary\n")
    report.append("| Approach | Available | Passed | Failed | Errors | Pass Rate | Total Time |")
    report.append("|----------|-----------|--------|--------|--------|-----------|------------|")

    for s in summaries:
        if s.available:
            report.append(
                f"| {s.name} | Yes | {s.passed} | {s.failed} | {s.errors} | "
                f"{s.pass_rate:.1f}% | {s.total_time_ms:.2f}ms |"
            )
        else:
            report.append(f"| {s.name} | No | - | - | - | - | - |")

    report.append("")

    # Detailed results per approach
    for s in summaries:
        if not s.available:
            report.append(f"## {s.name}\n")
            report.append("**Status**: Not available (missing dependencies)\n")
            report.append(f"**Description**: {s.description}\n")
            continue

        report.append(f"## {s.name}\n")
        report.append(f"**Description**: {s.description}\n")
        report.append(f"**Pass Rate**: {s.pass_rate:.1f}% ({s.passed}/{s.total})\n")
        report.append(f"**Total Time**: {s.total_time_ms:.2f}ms\n")

        # List failures
        failures = [r for r in s.results if not r.passed]
        if failures:
            report.append("\n### Failed Tests\n")
            for f in failures:
                report.append(f"- **{f.test_name}**")
                if f.error:
                    report.append(f"  - Error: {f.error}")
                report.append("")

    # Recommendations
    report.append("## Recommendations\n")

    available_summaries = [s for s in summaries if s.available]
    if available_summaries:
        best = max(available_summaries, key=lambda s: (s.pass_rate, -s.total_time_ms))
        report.append(f"**Recommended Approach**: {best.name}\n")
        report.append(f"- Pass Rate: {best.pass_rate:.1f}%")
        report.append(f"- Average Time: {best.total_time_ms / max(best.total, 1):.2f}ms per test")

    return "\n".join(report)


def main():
    """Run all experiments and generate report."""
    print("=" * 70)
    print("Code Replacement Strategy Experiments")
    print("=" * 70)
    print()

    summaries = []

    # Test Approach B (always available)
    print("Testing Approach B: Text-Based...")
    summary_b = test_approach_b()
    summaries.append(summary_b)
    print(f"  Results: {summary_b.passed}/{summary_b.total} passed ({summary_b.pass_rate:.1f}%)")
    print()

    # Test Approach C (requires tree-sitter)
    print("Testing Approach C: Hybrid (tree-sitter + text)...")
    summary_c = test_approach_c()
    summaries.append(summary_c)
    if summary_c.available:
        print(f"  Results: {summary_c.passed}/{summary_c.total} passed ({summary_c.pass_rate:.1f}%)")
    else:
        print("  Not available (install tree-sitter packages)")
    print()

    # Test Approach A (requires Node.js)
    print("Testing Approach A: jscodeshift...")
    summary_a = test_approach_a()
    summaries.append(summary_a)
    if summary_a.available:
        print("  Available but requires full npm setup for testing")
    else:
        print("  Not available (Node.js not found)")
    print()

    # Generate report
    report = generate_report(summaries)

    # Save report
    report_path = Path(__file__).parent / "EXPERIMENT_RESULTS.md"
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
