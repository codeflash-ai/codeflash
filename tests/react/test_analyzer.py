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


class TestDetectEagerStateInit:
    def test_eager_function_call_in_usestate(self):
        source = (
            "function TestComponent() {\n"
            "  const [data, setData] = useState(computeInitialData());\n"
            "  return <div>{data}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EAGER_STATE_INIT in types

    def test_lazy_initializer_not_flagged(self):
        source = (
            "function TestComponent() {\n"
            "  const [data, setData] = useState(() => computeInitialData());\n"
            "  return <div>{data}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        eager_opps = [o for o in opps if o.type == OpportunityType.EAGER_STATE_INIT]
        assert len(eager_opps) == 0

    def test_literal_not_flagged(self):
        source = (
            "function TestComponent() {\n"
            "  const [count, setCount] = useState(0);\n"
            "  const [name, setName] = useState('hello');\n"
            "  const [flag, setFlag] = useState(true);\n"
            "  return <div>{count}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=6)
        opps = detect_optimization_opportunities(source, info)
        eager_opps = [o for o in opps if o.type == OpportunityType.EAGER_STATE_INIT]
        assert len(eager_opps) == 0

    def test_eager_with_typescript_generic(self):
        source = (
            "function TestComponent() {\n"
            "  const [items, setItems] = useState<Item[]>(getDefaultItems());\n"
            "  return <div>{items.length}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EAGER_STATE_INIT in types


class TestDetectExpensiveRenderCalls:
    def test_new_regexp_in_render(self):
        source = (
            "function TestComponent({ pattern }) {\n"
            "  const regex = new RegExp(pattern, 'i');\n"
            "  return <div>{regex.test('hello') ? 'match' : 'no match'}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EXPENSIVE_RENDER_CALL in types

    def test_new_date_in_render(self):
        source = (
            "function TestComponent() {\n"
            "  const now = new Date();\n"
            "  return <div>{now.toISOString()}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EXPENSIVE_RENDER_CALL in types

    def test_json_parse_in_render(self):
        source = (
            "function TestComponent({ raw }) {\n"
            "  const data = JSON.parse(raw);\n"
            "  return <div>{data.name}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EXPENSIVE_RENDER_CALL in types

    def test_new_map_in_render(self):
        source = (
            "function TestComponent({ entries }) {\n"
            "  const lookup = new Map(entries);\n"
            "  return <div>{lookup.size}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.EXPENSIVE_RENDER_CALL in types

    def test_usememo_wrapped_not_flagged(self):
        source = (
            "function TestComponent({ pattern }) {\n"
            "  const regex = useMemo(() => new RegExp(pattern, 'i'), [pattern]);\n"
            "  return <div>{regex.test('hello') ? 'match' : 'no match'}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        expensive_opps = [o for o in opps if o.type == OpportunityType.EXPENSIVE_RENDER_CALL]
        assert len(expensive_opps) == 0


class TestDetectCombinableLoops:
    def test_triple_iteration_on_same_var(self):
        source = (
            "function TestComponent({ items }) {\n"
            "  const active = items.filter(i => i.active);\n"
            "  const names = items.map(i => i.name);\n"
            "  const total = items.reduce((s, i) => s + i.value, 0);\n"
            "  return <div>{total}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=6)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.COMBINABLE_LOOPS in types

    def test_two_iterations_not_flagged(self):
        source = (
            "function TestComponent({ items }) {\n"
            "  const active = items.filter(i => i.active);\n"
            "  const total = items.reduce((s, i) => s + i.value, 0);\n"
            "  return <div>{total}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=5)
        opps = detect_optimization_opportunities(source, info)
        combinable = [o for o in opps if o.type == OpportunityType.COMBINABLE_LOOPS]
        assert len(combinable) == 0

    def test_different_vars_not_flagged(self):
        source = (
            "function TestComponent({ users, products }) {\n"
            "  const activeUsers = users.filter(u => u.active);\n"
            "  const expensiveProducts = products.filter(p => p.price > 100);\n"
            "  const cheapProducts = products.map(p => p.name);\n"
            "  return <div>{activeUsers.length}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=6)
        opps = detect_optimization_opportunities(source, info)
        combinable = [o for o in opps if o.type == OpportunityType.COMBINABLE_LOOPS]
        # users has 1 op, products has 2 ops — neither reaches 3
        assert len(combinable) == 0


class TestDetectSequentialAwaits:
    def test_two_sequential_awaits(self):
        source = (
            "function TestComponent() {\n"
            "  useEffect(() => {\n"
            "    async function load() {\n"
            "      const users = await fetchUsers();\n"
            "      const posts = await fetchPosts();\n"
            "      setData({ users, posts });\n"
            "    }\n"
            "    load();\n"
            "  }, []);\n"
            "  return <div />;\n"
            "}"
        )
        info = _make_component_info(end_line=11)
        opps = detect_optimization_opportunities(source, info)
        types = [o.type for o in opps]
        assert OpportunityType.SEQUENTIAL_AWAITS in types

    def test_promise_all_not_flagged(self):
        source = (
            "function TestComponent() {\n"
            "  useEffect(() => {\n"
            "    async function load() {\n"
            "      const [users, posts] = await Promise.all([fetchUsers(), fetchPosts()]);\n"
            "      setData({ users, posts });\n"
            "    }\n"
            "    load();\n"
            "  }, []);\n"
            "  return <div />;\n"
            "}"
        )
        info = _make_component_info(end_line=10)
        opps = detect_optimization_opportunities(source, info)
        sequential = [o for o in opps if o.type == OpportunityType.SEQUENTIAL_AWAITS]
        assert len(sequential) == 0

    def test_single_await_not_flagged(self):
        source = (
            "function TestComponent() {\n"
            "  useEffect(() => {\n"
            "    async function load() {\n"
            "      const data = await fetchData();\n"
            "      setData(data);\n"
            "    }\n"
            "    load();\n"
            "  }, []);\n"
            "  return <div />;\n"
            "}"
        )
        info = _make_component_info(end_line=10)
        opps = detect_optimization_opportunities(source, info)
        sequential = [o for o in opps if o.type == OpportunityType.SEQUENTIAL_AWAITS]
        assert len(sequential) == 0


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

    def test_eager_state_init_is_high(self):
        source = (
            "function TestComponent() {\n"
            "  const [data] = useState(expensiveInit());\n"
            "  return <div>{data}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        eager_opps = [o for o in opps if o.type == OpportunityType.EAGER_STATE_INIT]
        assert all(o.severity == OpportunitySeverity.HIGH for o in eager_opps)

    def test_expensive_render_call_is_high(self):
        source = (
            "function TestComponent({ pattern }) {\n"
            "  const regex = new RegExp(pattern);\n"
            "  return <div>{String(regex)}</div>;\n"
            "}"
        )
        info = _make_component_info(end_line=4)
        opps = detect_optimization_opportunities(source, info)
        expensive_opps = [o for o in opps if o.type == OpportunityType.EXPENSIVE_RENDER_CALL]
        assert all(o.severity == OpportunitySeverity.HIGH for o in expensive_opps)
