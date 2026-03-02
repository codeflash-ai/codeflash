"""End-to-end integration test for the React optimization pipeline.

Tests the full flow: framework detection → component discovery → context extraction
→ profiler marker parsing → benchmarking → critic evaluation.

Note: This does not invoke the LLM or run actual Jest tests. It validates the
pipeline wiring by running each stage on fixture files and verifying outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "react" / "fixtures"


@pytest.fixture(autouse=True)
def clear_framework_cache():
    from codeflash.languages.javascript.frameworks.detector import detect_framework

    detect_framework.cache_clear()
    yield
    detect_framework.cache_clear()


class TestReactPipelineE2E:
    def test_framework_detection_from_fixture(self):
        from codeflash.languages.javascript.frameworks.detector import detect_framework

        info = detect_framework(FIXTURES_DIR)
        assert info.name == "react"
        assert info.react_version_major == 18
        assert info.has_testing_library is True

    def test_component_discovery(self):
        from codeflash.languages.javascript.frameworks.react.discovery import find_react_components
        from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer("tsx")

        # Counter.tsx — should find 1 function component
        source = FIXTURES_DIR.joinpath("Counter.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "Counter.tsx", analyzer)
        assert any(c.function_name == "Counter" for c in components)

        # ServerComponent.tsx — should be skipped (use server)
        source = FIXTURES_DIR.joinpath("ServerComponent.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "ServerComponent.tsx", analyzer)
        assert components == []

        # useDebounce.ts — should be detected as hook, not component
        ts_analyzer = TreeSitterAnalyzer("typescript")
        source = FIXTURES_DIR.joinpath("useDebounce.ts").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "useDebounce.ts", ts_analyzer)
        hooks = [c for c in components if c.component_type.value == "hook"]
        assert len(hooks) == 1
        assert hooks[0].function_name == "useDebounce"

    def test_optimization_opportunity_detection(self):
        from codeflash.languages.javascript.frameworks.react.analyzer import (
            OpportunityType,
            detect_optimization_opportunities,
        )
        from codeflash.languages.javascript.frameworks.react.discovery import (
            ComponentType,
            ReactComponentInfo,
            find_react_components,
        )
        from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer("tsx")

        # DataTable has expensive operations without useMemo
        source = FIXTURES_DIR.joinpath("DataTable.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "DataTable.tsx", analyzer)
        data_table = [c for c in components if c.function_name == "DataTable"][0]
        opps = detect_optimization_opportunities(source, data_table)
        opp_types = [o.type for o in opps]
        assert OpportunityType.MISSING_USEMEMO in opp_types

        # UserCard has inline objects in JSX
        source = FIXTURES_DIR.joinpath("UserCard.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "UserCard.tsx", analyzer)
        user_card = [c for c in components if c.function_name == "UserCard"][0]
        opps = detect_optimization_opportunities(source, user_card)
        opp_types = [o.type for o in opps]
        assert OpportunityType.INLINE_OBJECT_PROP in opp_types

    def test_context_extraction(self):
        from codeflash.languages.javascript.frameworks.react.context import extract_react_context
        from codeflash.languages.javascript.frameworks.react.discovery import find_react_components
        from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer("tsx")
        source = FIXTURES_DIR.joinpath("TaskList.tsx").read_text(encoding="utf-8")
        components = find_react_components(source, FIXTURES_DIR / "TaskList.tsx", analyzer)
        task_list = [c for c in components if c.function_name == "TaskList"][0]

        context = extract_react_context(task_list, source, analyzer, FIXTURES_DIR)
        assert len(context.hooks_used) > 0
        assert len(context.optimization_opportunities) > 0

        prompt = context.to_prompt_string()
        assert "useState" in prompt or "Hooks used" in prompt

    def test_profiler_marker_parsing(self):
        from codeflash.languages.javascript.parse import parse_react_render_markers

        stdout = (
            "PASS src/TaskList.test.tsx\n"
            "!######REACT_RENDER:TaskList:mount:25.3:40.1:1######!\n"
            "!######REACT_RENDER:TaskList:update:5.2:40.1:5######!\n"
            "!######REACT_RENDER:TaskList:update:4.8:40.1:10######!\n"
        )
        profiles = parse_react_render_markers(stdout)
        assert len(profiles) == 3
        assert profiles[0].component_name == "TaskList"
        assert profiles[2].render_count == 10

    def test_benchmarking_and_critic(self):
        from codeflash.languages.javascript.frameworks.react.benchmarking import (
            compare_render_benchmarks,
            format_render_benchmark_for_pr,
        )
        from codeflash.languages.javascript.parse import RenderProfile
        from codeflash.result.critic import render_efficiency_critic

        original = [
            RenderProfile("TaskList", "mount", 25.0, 40.0, 1),
            RenderProfile("TaskList", "update", 5.0, 40.0, 25),
            RenderProfile("TaskList", "update", 4.5, 40.0, 47),
        ]
        optimized = [
            RenderProfile("TaskList", "mount", 20.0, 35.0, 1),
            RenderProfile("TaskList", "update", 2.0, 35.0, 3),
        ]

        benchmark = compare_render_benchmarks(original, optimized)
        assert benchmark is not None
        assert benchmark.original_render_count == 47
        assert benchmark.optimized_render_count == 3
        assert benchmark.render_count_reduction_pct > 90

        # Critic should accept this optimization
        accepted = render_efficiency_critic(
            original_render_count=benchmark.original_render_count,
            optimized_render_count=benchmark.optimized_render_count,
            original_render_duration=benchmark.original_avg_duration_ms,
            optimized_render_duration=benchmark.optimized_avg_duration_ms,
        )
        assert accepted is True

        # PR formatting
        pr_output = format_render_benchmark_for_pr(benchmark)
        assert "47" in pr_output
        assert "3" in pr_output
        assert "React Render Performance" in pr_output