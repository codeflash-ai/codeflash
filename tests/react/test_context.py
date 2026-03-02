"""Tests for React context extraction."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.context import (
    HookUsage,
    ReactContext,
    _extract_child_components,
    _extract_context_subscriptions,
    _extract_hook_usages,
)


class TestExtractHookUsages:
    def test_useState_no_deps(self):
        source = "const [count, setCount] = useState(0);"
        hooks = _extract_hook_usages(source)
        assert len(hooks) == 1
        assert hooks[0].name == "useState"
        assert hooks[0].has_dependency_array is False

    def test_useEffect_with_deps(self):
        source = "useEffect(() => { fetchData(); }, [id, query]);"
        hooks = _extract_hook_usages(source)
        assert len(hooks) == 1
        assert hooks[0].name == "useEffect"
        assert hooks[0].has_dependency_array is True
        assert hooks[0].dependency_count == 2

    def test_useEffect_empty_deps(self):
        source = "useEffect(() => { setup(); }, []);"
        hooks = _extract_hook_usages(source)
        assert len(hooks) == 1
        assert hooks[0].has_dependency_array is True
        assert hooks[0].dependency_count == 0

    def test_useMemo_with_deps(self):
        source = "const value = useMemo(() => compute(a, b), [a, b]);"
        hooks = _extract_hook_usages(source)
        assert len(hooks) == 1
        assert hooks[0].name == "useMemo"
        assert hooks[0].has_dependency_array is True
        assert hooks[0].dependency_count == 2

    def test_multiple_hooks(self):
        source = """
        const [count, setCount] = useState(0);
        const ref = useRef(null);
        useEffect(() => {}, [count]);
        """
        hooks = _extract_hook_usages(source)
        names = [h.name for h in hooks]
        assert "useState" in names
        assert "useRef" in names
        assert "useEffect" in names

    def test_custom_hook(self):
        source = "const debouncedValue = useDebounce(value, 300);"
        hooks = _extract_hook_usages(source)
        assert len(hooks) == 1
        assert hooks[0].name == "useDebounce"


class TestExtractChildComponents:
    def test_finds_child_components(self):
        source = """
        return (
            <div>
                <Header />
                <Sidebar items={items} />
                <MainContent>
                    <Card title="test" />
                </MainContent>
            </div>
        );
        """
        # We pass a dummy analyzer — child extraction uses regex
        children = _extract_child_components(source, None, source)
        assert "Header" in children
        assert "Sidebar" in children
        assert "MainContent" in children
        assert "Card" in children

    def test_excludes_react_builtins(self):
        source = """
        return (
            <React.Fragment>
                <Suspense fallback={<div/>}>
                    <MyComponent />
                </Suspense>
            </React.Fragment>
        );
        """
        children = _extract_child_components(source, None, source)
        assert "React.Fragment" not in children
        assert "Fragment" not in children
        assert "Suspense" not in children
        assert "MyComponent" in children

    def test_html_elements_excluded(self):
        source = """
        return (
            <div>
                <span>text</span>
                <button onClick={fn}>click</button>
            </div>
        );
        """
        children = _extract_child_components(source, None, source)
        assert len(children) == 0


class TestExtractContextSubscriptions:
    def test_useContext_call(self):
        source = "const theme = useContext(ThemeContext);"
        subs = _extract_context_subscriptions(source)
        assert subs == ["ThemeContext"]

    def test_multiple_contexts(self):
        source = """
        const user = useContext(UserContext);
        const theme = useContext(ThemeContext);
        """
        subs = _extract_context_subscriptions(source)
        assert "UserContext" in subs
        assert "ThemeContext" in subs

    def test_no_context(self):
        source = "const [count, setCount] = useState(0);"
        subs = _extract_context_subscriptions(source)
        assert subs == []


class TestReactContextPromptString:
    def test_full_context(self):
        ctx = ReactContext(
            props_interface="interface Props { name: string; }",
            hooks_used=[HookUsage(name="useState", has_dependency_array=False, dependency_count=0)],
            child_components=["Header", "Footer"],
            context_subscriptions=["ThemeContext"],
            is_already_memoized=False,
        )
        prompt = ctx.to_prompt_string()
        assert "Props" in prompt
        assert "useState" in prompt
        assert "Header" in prompt
        assert "ThemeContext" in prompt

    def test_memoized_note(self):
        ctx = ReactContext(is_already_memoized=True)
        prompt = ctx.to_prompt_string()
        assert "React.memo()" in prompt

    def test_empty_context(self):
        ctx = ReactContext()
        prompt = ctx.to_prompt_string()
        assert prompt == ""
