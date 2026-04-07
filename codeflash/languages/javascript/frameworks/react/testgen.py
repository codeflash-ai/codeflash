"""React-specific test generation helpers.

Provides context building for React testgen prompts, re-render counting
test templates, and post-processing for generated React tests.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from codeflash.languages.base import CodeContext
    from codeflash.languages.javascript.frameworks.react.context import ReactContext
    from codeflash.languages.javascript.frameworks.react.discovery import ReactComponentInfo


def build_react_testgen_context(
    component_info: ReactComponentInfo, react_context: ReactContext, code_context: CodeContext
) -> dict[str, Any]:
    """Assemble context dict for the React testgen LLM prompt."""
    return {
        "component_name": component_info.function_name,
        "component_type": component_info.component_type.value,
        "component_source": code_context.target_code,
        "props_interface": react_context.props_interface or "",
        "hooks_used": [h.name for h in react_context.hooks_used],
        "child_components": react_context.child_components,
        "context_subscriptions": react_context.context_subscriptions,
        "is_memoized": component_info.is_memoized,
        "optimization_opportunities": [
            {"type": o.type.value, "line": o.line, "description": o.description}
            for o in react_context.optimization_opportunities
        ],
        "read_only_context": code_context.read_only_context,
        "imports": code_context.imports,
    }


def generate_rerender_test_template(component_name: str, props_interface: str | None = None) -> str:
    """Generate a template test that counts re-renders for a component.

    This template uses @testing-library/react's render + rerender to verify
    that same props don't cause unnecessary re-renders.
    """
    props_example = "{ /* same props */ }" if not props_interface else "{ /* fill in props matching interface */ }"

    return f"""\
import {{ render }} from '@testing-library/react';
import {{ {component_name} }} from './path-to-component';

describe('{component_name} render efficiency', () => {{
  it('should not re-render with same props', () => {{
    let renderCount = 0;
    const OriginalComponent = {component_name};

    // Wrap to count renders
    const CountingComponent = (props) => {{
      renderCount++;
      return <OriginalComponent {{...props}} />;
    }};

    const props = {props_example};
    const {{ rerender }} = render(<CountingComponent {{...props}} />);

    // Initial render
    expect(renderCount).toBe(1);

    // Re-render with same props
    rerender(<CountingComponent {{...props}} />);

    // Should not have re-rendered (if properly memoized)
    // For non-memoized components, renderCount will be 2
    console.log(`!######REACT_RENDER:{component_name}:rerender_test:0:0:${{renderCount}}######!`);
  }});

  it('should render correctly with props', () => {{
    const props = {props_example};
    const {{ container }} = render(<{component_name} {{...props}} />);
    expect(container).toBeTruthy();
  }});
}});
"""


def post_process_react_tests(test_source: str, component_info: ReactComponentInfo) -> str:
    """Post-process LLM-generated React tests.

    Ensures:
    - @testing-library/react imports are present
    - act() wrapping for state updates
    - cleanup import when unmount is used
    - Fake timers for debounce/throttle tests
    - user-event import for interaction tests
    """
    result = test_source

    # Fix outdated @testing-library/jest-dom import paths (v6+ removed /extend-expect subpath)
    result = result.replace("@testing-library/jest-dom/extend-expect", "@testing-library/jest-dom")
    # Also fix require() variant
    result = re.sub(
        r"""require\s*\(\s*['"]@testing-library/jest-dom/extend-expect['"]\s*\)""",
        "require('@testing-library/jest-dom')",
        result,
    )

    # Ensure testing-library import
    if "@testing-library/react" not in result:
        result = "import { render, screen, act } from '@testing-library/react';\n" + result

    # Ensure act is in the @testing-library/react import if act() is used in the test
    if "act(" in result:
        match = re.search(r"import\s*\{([^}]+)\}\s*from\s*['\"]@testing-library/react['\"]", result)
        if match:
            imports = match.group(1)
            if "act" not in imports:
                result = result.replace(match.group(0), match.group(0).replace("{" + imports + "}", "{" + imports + ", act}"))

    # Ensure cleanup import if unmount is called
    if "unmount" in result:
        match = re.search(r"import\s*\{([^}]+)\}\s*from\s*['\"]@testing-library/react['\"]", result)
        if match:
            imports = match.group(1)
            if "cleanup" not in imports:
                result = result.replace(match.group(0), match.group(0).replace("{" + imports + "}", "{" + imports + ", cleanup}"))

    # Ensure fake timers for debounce/throttle tests
    if re.search(r"jest\.advanceTimersByTime|jest\.runAllTimers|jest\.runOnlyPendingTimers|vi\.advanceTimersByTime", result):
        if "useFakeTimers" not in result:
            # Inject jest.useFakeTimers() at the start of beforeEach or before first describe
            before_each_match = re.search(r"(beforeEach\s*\(\s*(?:async\s*)?\(\)\s*=>\s*\{)", result)
            if before_each_match:
                result = result.replace(
                    before_each_match.group(0),
                    before_each_match.group(0) + "\n    jest.useFakeTimers();",
                )
                # Also add afterEach to restore real timers if not present
                if "useRealTimers" not in result:
                    after_each_match = re.search(r"(afterEach\s*\(\s*(?:async\s*)?\(\)\s*=>\s*\{)", result)
                    if after_each_match:
                        result = result.replace(
                            after_each_match.group(0),
                            after_each_match.group(0) + "\n    jest.useRealTimers();",
                        )
                    else:
                        # Add afterEach block after the beforeEach block
                        result = re.sub(
                            r"(beforeEach\s*\([^)]*\)\s*=>\s*\{[^}]*\}\s*\);?\s*\n)",
                            r"\1\n  afterEach(() => {\n    jest.useRealTimers();\n  });\n",
                            result,
                            count=1,
                        )
            else:
                # No beforeEach — inject before first describe/it block
                result = re.sub(
                    r"(describe\s*\()",
                    "beforeEach(() => {\n  jest.useFakeTimers();\n});\n\nafterEach(() => {\n  jest.useRealTimers();\n});\n\n\\1",
                    result,
                    count=1,
                )

    # Ensure user-event import if user interactions are tested
    if (
        "userEvent" in result or "user-event" in result
    ) and "@testing-library/user-event" not in result:
        result = re.sub(
            r"(import .+ from '@testing-library/react';?\n)",
            r"\1import userEvent from '@testing-library/user-event';\n",
            result,
            count=1,
        )

    # Auto-inject per-interaction render tracking markers around fireEvent/userEvent calls.
    # This gives per-interaction A/B signal without the LLM needing to know about it.
    result = inject_interaction_markers(result)

    # If no tests contain interaction calls, auto-inject a rerender fallback so
    # that EVERY React perf test produces at least one update-phase marker.
    if not has_react_test_interactions(result):
        logger.warning(
            "[REACT] Generated tests for %s contain no interactions — auto-injecting rerender fallback.",
            component_info.function_name,
        )
        result = _inject_rerender_fallback(result, component_info.function_name)

    # Check interaction density — fewer than MIN_INTERACTION_CALLS total interactions
    # means the test is unlikely to produce enough update-phase renders for reliable measurement.
    interaction_count = count_interaction_calls(result)
    if interaction_count < MIN_INTERACTION_CALLS:
        logger.error(
            "[REACT] Generated tests for %s have only %d interaction calls (minimum %d). "
            "Render count measurement will have low confidence.",
            component_info.function_name,
            interaction_count,
            MIN_INTERACTION_CALLS,
        )

    # Warn if tests lack high-density interaction patterns (loops or 3+ sequential calls)
    if not has_high_density_interactions(result):
        logger.warning(
            "[REACT] Generated tests for %s lack high-density interactions (no loops with interactions or "
            "3+ sequential interaction calls). Render count differences may be too small to measure.",
            component_info.function_name,
        )

    return result


# Pattern to find the variable assigned from captureRenderPerf (await or sync)
# Matches: const result = await codeflash.captureRenderPerf(...)
#          const { container } = await codeflash.captureRenderPerf(...)
#          let result = codeflash.captureRenderPerf(...)
_CAPTURE_RENDER_RESULT_PATTERN = re.compile(
    r"(?:const|let|var)\s+(?:\{[^}]+\}|(\w+))\s*=\s*(?:await\s+)?(?:\w+\.)?captureRenderPerf\(",
)

# Pattern matching fireEvent.* or userEvent.* standalone calls (not in comments)
_INTERACTION_CALL_PATTERN = re.compile(
    r"^(\s*)((?:await\s+)?(?:fireEvent\.\w+|userEvent\.\w+)\s*\([^)]*\))\s*;",
    re.MULTILINE,
)


def _extract_interaction_label(call_text: str) -> str:
    """Extract a short label from an interaction call, e.g. 'click' from 'fireEvent.click(...)'."""
    m = re.search(r"(?:fireEvent|userEvent)\.(\w+)", call_text)
    return m.group(1) if m else "interaction"


def inject_dom_snapshot_calls(test_source: str) -> str:
    """Inject codeflash.snapshotDOM() calls after each user interaction in behavior mode.

    Only active when `captureRender` (not `captureRenderPerf`) is present,
    meaning the test is running in behavioral verification mode.

    After each fireEvent.*, userEvent.*, or rerender() call, inserts:
        codeflash.snapshotDOM('after_{label}_{n}');
    on the next line with matching indentation.
    """
    if "captureRender" not in test_source:
        return test_source
    if "captureRenderPerf" in test_source:
        return test_source

    interaction_counter: dict[str, int] = {}
    lines = test_source.split("\n")
    new_lines: list[str] = []

    # Also match rerender() calls — use .* to handle nested parens like getByText('Add')
    snapshot_interaction_pattern = re.compile(
        r"^(\s*)((?:await\s+)?(?:fireEvent\.\w+|userEvent\.\w+|(?:\w+\.)?rerender)\s*\(.*\))\s*;?\s*$",
        re.MULTILINE,
    )

    for line in lines:
        new_lines.append(line)
        m = snapshot_interaction_pattern.match(line)
        if m:
            indent = m.group(1)
            call_text = m.group(2)
            label = _extract_interaction_label(call_text)
            if label == "interaction":
                # rerender() call
                label = "rerender"
            interaction_counter[label] = interaction_counter.get(label, 0) + 1
            unique_label = f"{label}_{interaction_counter[label]}"
            new_lines.append(f"{indent}codeflash.snapshotDOM('after_{unique_label}');")

    return "\n".join(new_lines)


def inject_interaction_markers(test_source: str) -> str:
    """Inject _codeflashMarkInteraction() calls before each fireEvent/userEvent call.

    Only injects when captureRenderPerf is used (the result object has the method).
    Assigns a label derived from the interaction type (click, change, type, etc.)
    and a sequential counter for uniqueness.
    """
    if "captureRenderPerf" not in test_source:
        return test_source

    # Find the result variable name from captureRenderPerf assignment
    # Support both: const result = ... and const { container, ...rest } = ...
    result_var = None
    capture_match = _CAPTURE_RENDER_RESULT_PATTERN.search(test_source)
    if capture_match:
        # Group 1 is the simple variable name; for destructuring we need a different approach
        result_var = capture_match.group(1)
    if not result_var:
        # Look for destructuring pattern and use the first variable
        destr_match = re.search(
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:await\s+)?(?:\w+\.)?captureRenderPerf\(",
            test_source,
        )
        if destr_match:
            result_var = destr_match.group(1)
    if not result_var:
        # Can't determine result variable — skip injection
        return test_source

    # Find all interaction calls and inject marker before each
    interaction_counter: dict[str, int] = {}
    lines = test_source.split("\n")
    new_lines: list[str] = []
    for line in lines:
        m = _INTERACTION_CALL_PATTERN.match(line)
        if m:
            indent = m.group(1)
            call_text = m.group(2)
            label = _extract_interaction_label(call_text)
            interaction_counter[label] = interaction_counter.get(label, 0) + 1
            unique_label = f"{label}_{interaction_counter[label]}"
            marker_line = f"{indent}{result_var}._codeflashMarkInteraction('{unique_label}');"
            new_lines.append(marker_line)
        new_lines.append(line)

    return "\n".join(new_lines)


# Patterns that indicate a test triggers user interactions causing re-renders
_INTERACTION_PATTERNS = re.compile(
    r"fireEvent\.|userEvent\.|\.rerender\(|rerender\(|act\("
)


def has_react_test_interactions(test_source: str) -> bool:
    """Check if a React test contains interactions that trigger re-renders.

    Returns True if the test source contains any of: fireEvent, userEvent,
    rerender(), or act() calls — all of which cause update-phase renders
    that the Profiler can measure.
    """
    return bool(_INTERACTION_PATTERNS.search(test_source))


# Minimum interaction calls for reliable render count measurement
MIN_INTERACTION_CALLS = 3

# Pattern matching individual interaction calls (fireEvent.*, userEvent.*, .rerender(), rerender())
_INTERACTION_CALL_COUNT_PATTERN = re.compile(
    r"(?:fireEvent\.\w+|userEvent\.\w+|\.rerender\(|(?<!\.)rerender\()\s*\(",
)


def count_interaction_calls(test_source: str) -> int:
    """Count the number of interaction calls in a test source.

    Counts fireEvent.*, userEvent.*, and rerender() calls. Used to assess
    whether tests produce enough update-phase renders for reliable measurement.
    """
    return len(_INTERACTION_CALL_COUNT_PATTERN.findall(test_source))


# Patterns for loops containing interaction calls
_LOOP_WITH_INTERACTION = re.compile(
    r"for\s*\([^)]*\)\s*\{[^}]*(?:fireEvent\.|userEvent\.|rerender\()",
    re.DOTALL,
)

# Minimum sequential interaction calls to consider "high density"
_MIN_SEQUENTIAL_INTERACTIONS = 3


def has_high_density_interactions(test_source: str) -> bool:
    """Check if tests contain high-density interaction patterns.

    Returns True if the test source contains either:
    - A loop (for/while) with interaction calls inside, OR
    - 3+ sequential interaction calls (fireEvent/userEvent/rerender)
    """
    if _LOOP_WITH_INTERACTION.search(test_source):
        return True

    interaction_calls = _INTERACTION_PATTERNS.findall(test_source)
    return len(interaction_calls) >= _MIN_SEQUENTIAL_INTERACTIONS


# Pattern to extract props from the first render(<Component ...props... />) call
_RENDER_CALL_PROPS_PATTERN = re.compile(
    r"render\s*\(\s*<(\w+)\s+([^/]*?)\s*/?\s*>",
)


def _extract_render_props(test_source: str, component_name: str) -> str | None:
    """Extract the props expression from the first render(<Component ...>) call."""
    for m in _RENDER_CALL_PROPS_PATTERN.finditer(test_source):
        if m.group(1) == component_name:
            props_text = m.group(2).strip()
            if props_text:
                return props_text
    return None


def _inject_rerender_fallback(test_source: str, component_name: str) -> str:
    """Inject a rerender efficiency test block when the test has no interactions.

    This ensures every React perf test produces at least one update-phase marker.
    """
    # Try to extract props from existing render call
    props_expr = _extract_render_props(test_source, component_name)
    if props_expr:
        jsx_open = f"<{component_name} {props_expr} />"
    else:
        jsx_open = f"<{component_name} />"

    rerender_block = f"""
describe('{component_name} rerender efficiency (auto-generated)', () => {{
  it('should handle same-props rerenders', () => {{
    const {{ rerender }} = render({jsx_open});
    for (let i = 0; i < 10; i++) {{
      rerender({jsx_open});
    }}
  }});
}});
"""
    # Ensure {rerender} is in the @testing-library/react import
    rtl_import_match = re.search(
        r"import\s*\{([^}]+)\}\s*from\s*['\"]@testing-library/react['\"]", test_source
    )
    if rtl_import_match:
        imports = rtl_import_match.group(1)
        if "rerender" not in imports:
            # Don't add 'rerender' to import — it comes from render() return value, not an import
            pass

    return test_source.rstrip() + "\n" + rerender_block
