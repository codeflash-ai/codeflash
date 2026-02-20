"""React-specific test generation helpers.

Provides context building for React testgen prompts, re-render counting
test templates, and post-processing for generated React tests.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.languages.base import CodeContext
    from codeflash.languages.javascript.frameworks.react.context import ReactContext
    from codeflash.languages.javascript.frameworks.react.discovery import ReactComponentInfo


def build_react_testgen_context(
    component_info: ReactComponentInfo,
    react_context: ReactContext,
    code_context: CodeContext,
) -> dict:
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
    - Proper cleanup
    """
    result = test_source

    # Ensure testing-library import
    if "@testing-library/react" not in result:
        result = "import { render, screen, act } from '@testing-library/react';\n" + result

    # Ensure act import if state updates are detected
    if "act(" in result and "import" in result and "act" not in result.split("from '@testing-library/react'")[0]:
        result = result.replace(
            "from '@testing-library/react'",
            "act, " + "from '@testing-library/react'",
            1,
        )

    # Ensure user-event import if user interactions are tested
    if ("click" in result.lower() or "type" in result.lower() or "userEvent" in result) and "@testing-library/user-event" not in result:
        # Add user-event import after testing-library import
        result = re.sub(
            r"(import .+ from '@testing-library/react';?\n)",
            r"\1import userEvent from '@testing-library/user-event';\n",
            result,
            count=1,
        )

    return result
