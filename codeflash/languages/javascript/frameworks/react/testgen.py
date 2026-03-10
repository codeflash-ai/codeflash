"""React-specific test generation helpers.

Provides context building for React testgen prompts, re-render counting
test templates, and post-processing for generated React tests.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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

    return result
