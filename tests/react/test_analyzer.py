"""Tests for React optimization opportunity detection."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.analyzer import (
    OpportunitySeverity,
    OpportunityType,
    detect_optimization_opportunities,
)
from codeflash.languages.javascript.frameworks.react.discovery import (
    ComponentType,
    ReactComponentInfo,
)


def _make_component_info(
    name: str = "TestComponent",
    start_line: int = 1,
    end_line: int = 20,
    is_memoized: bool = False,
) -> ReactComponentInfo:
    return ReactComponentInfo(
        function_name=name,
        component_type=ComponentType.FUNCTION,
        returns_jsx=True,
        is_memoized=is_memoized,
        start_line=start_line,
        end_line=end_line,
    )


class TestDetectInlineObjects:
    def test_inline_style_prop(self):
        source = 'function TestComponent() {\n  return <div style={{ color: "red" }}>hello</div>;\n}'
        info = _make_component_info(end_line=3)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.INLINE_OBJECT_PROP in types

    def test_inline_array_prop(self):
        source = "function TestComponent() {\n  return <Select options={['a', 'b']} />;\n}"
        info = _make_component_info(end_line=3)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.INLINE_ARRAY_PROP in types

    def test_no_inline_props(self):
        source = "function TestComponent() {\n  const style = { color: 'red' };\n  return <div style={style}>hello</div>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        inline_opps = [o for o in opps if o.type in (OpportunityType.INLINE_OBJECT_PROP, OpportunityType.INLINE_ARRAY_PROP)]
        assert len(inline_opps) == 0


class TestDetectMissingUseCallback:
    def test_arrow_function_in_render(self):
        source = "function TestComponent() {\n  const handleClick = () => console.log('click');\n  return <button onClick={handleClick}>Click</button>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_USECALLBACK in types

    def test_function_expression_in_render(self):
        source = "function TestComponent() {\n  function handleSubmit(e) { e.preventDefault(); }\n  return <form onSubmit={handleSubmit}></form>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_USECALLBACK in types

    def test_usecallback_present_not_flagged(self):
        source = "function TestComponent() {\n  const handleClick = useCallback(() => console.log('click'), []);\n  return <button onClick={handleClick}>Click</button>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        callback_opps = [o for o in opps if o.type == OpportunityType.MISSING_USECALLBACK]
        assert len(callback_opps) == 0


class TestDetectMissingUseMemo:
    def test_filter_in_render(self):
        source = "function TestComponent({ items }) {\n  const filtered = items.filter(i => i.active);\n  return <ul>{filtered.map(i => <li key={i.id}>{i.name}</li>)}</ul>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_USEMEMO in types

    def test_sort_in_render(self):
        source = "function TestComponent({ items }) {\n  const sorted = items.sort((a, b) => a.name.localeCompare(b.name));\n  return <ul>{sorted.map(i => <li key={i.id}>{i.name}</li>)}</ul>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_USEMEMO in types

    def test_reduce_in_render(self):
        source = "function TestComponent({ items }) {\n  const total = items.reduce((sum, i) => sum + i.value, 0);\n  return <div>{total}</div>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_USEMEMO in types

    def test_usememo_present_not_flagged(self):
        source = "function TestComponent({ items }) {\n  const filtered = useMemo(() => items.filter(i => i.active), [items]);\n  return <div>{filtered.length} items</div>;\n}"
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        memo_opps = [o for o in opps if o.type == OpportunityType.MISSING_USEMEMO]
        assert len(memo_opps) == 0


class TestDetectMissingReactMemo:
    def test_non_memoized_component(self):
        source = "function TestComponent() {\n  return <div>hello</div>;\n}"
        info = _make_component_info(is_memoized=False, end_line=3)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.MISSING_REACT_MEMO in types

    def test_memoized_component_not_flagged(self):
        source = "function TestComponent() {\n  return <div>hello</div>;\n}"
        info = _make_component_info(is_memoized=True, end_line=3)
        opps = detect_optimization_opportunities(source, info)
        memo_opps = [o for o in opps if o.type == OpportunityType.MISSING_REACT_MEMO]
        assert len(memo_opps) == 0


class TestSeverityLevels:
    def test_inline_object_is_high(self):
        source = 'function TestComponent() {\n  return <div style={{ color: "red" }}>hello</div>;\n}'
        info = _make_component_info(end_line=3)
        opps = detect_optimization_opportunities(source, info)
        inline_opps = [o for o in opps if o.type == OpportunityType.INLINE_OBJECT_PROP]
        assert all(o.severity == OpportunitySeverity.HIGH for o in inline_opps)

    def test_missing_memo_is_medium(self):
        source = "function TestComponent() {\n  return <div>hello</div>;\n}"
        info = _make_component_info(is_memoized=False, end_line=3)
        opps = detect_optimization_opportunities(source, info)
        memo_opps = [o for o in opps if o.type == OpportunityType.MISSING_REACT_MEMO]
        assert all(o.severity == OpportunitySeverity.MEDIUM for o in memo_opps)
