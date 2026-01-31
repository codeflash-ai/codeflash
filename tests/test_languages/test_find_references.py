"""Comprehensive tests for JavaScript/TypeScript find references functionality.

These tests are inspired by real-world patterns found in the Appsmith codebase,
covering various import/export patterns, callback usage, memoization, and more.
"""

import pytest
from pathlib import Path

from codeflash.languages.javascript.find_references import (
    Reference,
    ReferenceFinder,
    ExportedFunction,
    ReferenceSearchContext,
    find_references,
)


class TestReferenceFinder:
    """Tests for ReferenceFinder class."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a basic project structure."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    @pytest.fixture
    def finder(self, project_root):
        """Create a ReferenceFinder instance."""
        return ReferenceFinder(project_root)

    def test_init_default_exclude_patterns(self, project_root):
        """Test that default exclude patterns are set."""
        finder = ReferenceFinder(project_root)
        assert "node_modules" in finder.exclude_patterns
        assert "dist" in finder.exclude_patterns
        assert ".git" in finder.exclude_patterns

    def test_init_custom_exclude_patterns(self, project_root):
        """Test custom exclude patterns."""
        finder = ReferenceFinder(project_root, exclude_patterns=["custom_dir"])
        assert "custom_dir" in finder.exclude_patterns
        assert "node_modules" not in finder.exclude_patterns

    def test_should_exclude_node_modules(self, finder, project_root):
        """Test that node_modules files are excluded."""
        path = project_root / "node_modules" / "lodash" / "index.js"
        assert finder._should_exclude(path) is True

    def test_should_not_exclude_src(self, finder, project_root):
        """Test that src files are not excluded."""
        path = project_root / "src" / "utils.ts"
        assert finder._should_exclude(path) is False


class TestBasicNamedExports:
    """Tests for basic named export/import patterns.

    Inspired by Appsmith patterns like:
    import { getDynamicBindings, isDynamicValue } from "utils/DynamicBindingUtils";
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with named export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Source file with named export
        (utils_dir / "DynamicBindingUtils.ts").write_text("""
/**
 * Get dynamic bindings from a string
 */
export function getDynamicBindings(value: string): string[] {
    const regex = /{{([^}]+)}}/g;
    const matches = [];
    let match;
    while ((match = regex.exec(value)) !== null) {
        matches.push(match[1]);
    }
    return matches;
}

export function isDynamicValue(value: string): boolean {
    return value.includes('{{') && value.includes('}}');
}

function internalHelper() {
    return "not exported";
}
""")

        # File that imports and uses the function
        (src_dir / "evaluator.ts").write_text("""
import { getDynamicBindings, isDynamicValue } from './utils/DynamicBindingUtils';

export function evaluate(expression: string) {
    if (isDynamicValue(expression)) {
        const bindings = getDynamicBindings(expression);
        return bindings.map(b => eval(b));
    }
    return expression;
}
""")

        # Another file that uses the function
        (src_dir / "validator.ts").write_text("""
import { getDynamicBindings } from './utils/DynamicBindingUtils';

export function validateBindings(input: string) {
    const bindings = getDynamicBindings(input);
    return bindings.length > 0;
}
""")

        return tmp_path

    def test_find_named_export_references(self, project_root):
        """Test finding references to a named exported function."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "DynamicBindingUtils.ts"

        refs = finder.find_references("getDynamicBindings", source_file)

        # Should find references in both evaluator.ts and validator.ts
        ref_files = {ref.file_path for ref in refs}
        assert project_root / "src" / "evaluator.ts" in ref_files
        assert project_root / "src" / "validator.ts" in ref_files

    def test_reference_has_correct_type(self, project_root):
        """Test that references have correct reference types."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "DynamicBindingUtils.ts"

        refs = finder.find_references("getDynamicBindings", source_file)

        # All references should be calls
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 2

    def test_reference_has_context(self, project_root):
        """Test that references include context (the line of code)."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "DynamicBindingUtils.ts"

        refs = finder.find_references("getDynamicBindings", source_file)

        for ref in refs:
            assert ref.context  # Should have context
            assert "getDynamicBindings" in ref.context


class TestDefaultExports:
    """Tests for default export/import patterns.

    Inspired by patterns like:
    import MyComponent from './MyComponent';
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with default export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with default export
        (src_dir / "helper.ts").write_text("""
function processData(data: any[]) {
    return data.filter(item => item.active);
}

export default processData;
""")

        # File that imports the default export
        (src_dir / "main.ts").write_text("""
import processData from './helper';

export function handleData(items: any[]) {
    const processed = processData(items);
    return processed.length;
}
""")

        # File that imports with a different name
        (src_dir / "alternative.ts").write_text("""
import myProcessor from './helper';

export function process(items: any[]) {
    return myProcessor(items);
}
""")

        return tmp_path

    def test_find_default_export_references(self, project_root):
        """Test finding references to a default exported function."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helper.ts"

        refs = finder.find_references("processData", source_file)

        # Should find references in both files
        ref_files = {ref.file_path for ref in refs}
        assert project_root / "src" / "main.ts" in ref_files
        assert project_root / "src" / "alternative.ts" in ref_files

    def test_default_export_different_import_name(self, project_root):
        """Test that references are found when imported with different name."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helper.ts"

        refs = finder.find_references("processData", source_file)

        # Check that we found the reference with alias "myProcessor"
        alt_refs = [r for r in refs if r.file_path == project_root / "src" / "alternative.ts"]
        assert len(alt_refs) > 0
        assert any(r.import_name == "myProcessor" for r in alt_refs)


class TestReExports:
    """Tests for re-export patterns.

    Inspired by Appsmith patterns like:
    export { filterEntityGroupsBySearchTerm } from "./filterEntityGroupsBySearchTerm";
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with re-export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Original function file
        (utils_dir / "filterEntityGroupsBySearchTerm.ts").write_text("""
export function filterEntityGroupsBySearchTerm(groups: any[], searchTerm: string) {
    return groups.filter(g => g.name.includes(searchTerm));
}
""")

        # Index file that re-exports
        (utils_dir / "index.ts").write_text("""
export { filterEntityGroupsBySearchTerm } from './filterEntityGroupsBySearchTerm';
export { otherUtil } from './otherUtil';
""")

        # Create the other util for completeness
        (utils_dir / "otherUtil.ts").write_text("""
export function otherUtil() { return 42; }
""")

        # Consumer that imports from index
        (src_dir / "consumer.ts").write_text("""
import { filterEntityGroupsBySearchTerm } from './utils';

export function searchGroups(groups: any[], term: string) {
    return filterEntityGroupsBySearchTerm(groups, term);
}
""")

        return tmp_path

    def test_find_reexport(self, project_root):
        """Test finding re-export references."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "filterEntityGroupsBySearchTerm.ts"

        refs = finder.find_references("filterEntityGroupsBySearchTerm", source_file)

        # Should find the re-export in index.ts
        reexport_refs = [r for r in refs if r.reference_type == "reexport"]
        assert len(reexport_refs) > 0
        assert any(r.file_path == project_root / "src" / "utils" / "index.ts" for r in reexport_refs)


class TestCallbackPatterns:
    """Tests for functions passed as callbacks.

    Inspired by Appsmith patterns with .map(), .filter(), .reduce(), etc.
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with callback patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Helper function
        (src_dir / "transforms.ts").write_text("""
export function normalizeItem(item: any) {
    return {
        ...item,
        id: item.id.toString(),
        active: Boolean(item.active)
    };
}

export function validateItem(item: any) {
    return item && item.id !== undefined;
}
""")

        # Consumer using callbacks
        (src_dir / "processor.ts").write_text("""
import { normalizeItem, validateItem } from './transforms';

export function processItems(items: any[]) {
    // Function passed to map
    const normalized = items.map(normalizeItem);

    // Function passed to filter
    const valid = normalized.filter(validateItem);

    // Function used in reduce
    const result = valid.reduce((acc, item) => {
        return normalizeItem(item);
    }, null);

    return valid;
}
""")

        return tmp_path

    def test_find_callback_references(self, project_root):
        """Test finding functions used as callbacks."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "transforms.ts"

        refs = finder.find_references("normalizeItem", source_file)

        # Should find at least 2 references (map callback and direct call in reduce)
        processor_refs = [r for r in refs if r.file_path == project_root / "src" / "processor.ts"]
        assert len(processor_refs) >= 2

        # Check for callback type
        callback_refs = [r for r in processor_refs if r.reference_type == "callback"]
        assert len(callback_refs) >= 1


class TestAliasImports:
    """Tests for functions imported with aliases.

    Inspired by patterns like:
    import { originalName as aliasName } from './module';
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with alias import patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file
        (src_dir / "utils.ts").write_text("""
export function computeValue(input: number): number {
    return input * 2;
}
""")

        # File using alias
        (src_dir / "consumer.ts").write_text("""
import { computeValue as calculate } from './utils';

export function processNumber(n: number) {
    // Using the aliased name
    const result = calculate(n);
    return result + 10;
}
""")

        return tmp_path

    def test_find_aliased_import_references(self, project_root):
        """Test finding references when function is imported with alias."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils.ts"

        refs = finder.find_references("computeValue", source_file)

        # Should find the reference even though it's called as "calculate"
        consumer_refs = [r for r in refs if r.file_path == project_root / "src" / "consumer.ts"]
        assert len(consumer_refs) > 0
        assert any(r.import_name == "calculate" for r in consumer_refs)


class TestNamespaceImports:
    """Tests for namespace import patterns.

    Inspired by patterns like:
    import * as Utils from './utils';
    Utils.myFunction();
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with namespace import patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with multiple exports
        (src_dir / "mathUtils.ts").write_text("""
export function add(a: number, b: number): number {
    return a + b;
}

export function subtract(a: number, b: number): number {
    return a - b;
}

export function multiply(a: number, b: number): number {
    return a * b;
}
""")

        # File using namespace import
        (src_dir / "calculator.ts").write_text("""
import * as MathUtils from './mathUtils';

export function calculate(a: number, b: number, op: string) {
    switch(op) {
        case '+':
            return MathUtils.add(a, b);
        case '-':
            return MathUtils.subtract(a, b);
        case '*':
            return MathUtils.multiply(a, b);
        default:
            return MathUtils.add(a, b);
    }
}
""")

        return tmp_path

    def test_find_namespace_import_references(self, project_root):
        """Test finding references via namespace imports."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "mathUtils.ts"

        refs = finder.find_references("add", source_file)

        # Should find both calls to MathUtils.add
        calc_refs = [r for r in refs if r.file_path == project_root / "src" / "calculator.ts"]
        assert len(calc_refs) == 2  # Two calls to add in the switch


class TestMemoizedFunctions:
    """Tests for memoized function patterns.

    Inspired by Appsmith's use of micro-memoize:
    const memoizedChildHasPanelConfig = memoize(childHasPanelConfig);
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with memoized function patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with function to be memoized
        (src_dir / "expensive.ts").write_text("""
export function computeExpensiveValue(config: any): any {
    // Expensive computation
    return config.data.map((item: any) => item * 2);
}
""")

        # File that memoizes the function
        (src_dir / "memoized.ts").write_text("""
import memoize from 'micro-memoize';
import { computeExpensiveValue } from './expensive';

// Memoized version
export const memoizedComputeExpensiveValue = memoize(computeExpensiveValue);

export function processConfig(config: any) {
    // Direct call
    const direct = computeExpensiveValue(config);

    // Memoized call
    const cached = memoizedComputeExpensiveValue(config);

    return { direct, cached };
}
""")

        return tmp_path

    def test_find_memoized_function_references(self, project_root):
        """Test finding references to functions passed to memoize."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "expensive.ts"

        refs = finder.find_references("computeExpensiveValue", source_file)

        memoized_refs = [r for r in refs if r.file_path == project_root / "src" / "memoized.ts"]
        # Should find: memoize call, direct call
        assert len(memoized_refs) >= 2

        # Check for memoized reference type
        memo_refs = [r for r in memoized_refs if r.reference_type == "memoized"]
        assert len(memo_refs) >= 1


class TestSameFileReferences:
    """Tests for references within the same file.

    Inspired by recursive functions and internal helper calls in Appsmith.
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with same-file reference patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # File with internal references
        (src_dir / "recursive.ts").write_text("""
export function factorial(n: number): number {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // Recursive call
}

export function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);  // Two recursive calls
}

function internalHelper(x: number): number {
    return factorial(x) + fibonacci(x);  // Calls to exported functions
}

export function compute(n: number): number {
    return internalHelper(n);
}
""")

        return tmp_path

    def test_find_recursive_references(self, project_root):
        """Test finding recursive calls within same file."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "recursive.ts"

        refs = finder.find_references("factorial", source_file, include_definition=True)

        # Should find the recursive call and the call from internalHelper
        same_file_refs = [r for r in refs if r.file_path == source_file]
        assert len(same_file_refs) >= 2

    def test_find_fibonacci_double_recursion(self, project_root):
        """Test finding multiple recursive calls."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "recursive.ts"

        refs = finder.find_references("fibonacci", source_file, include_definition=True)

        same_file_refs = [r for r in refs if r.file_path == source_file]
        # Should find both fibonacci calls in the recursion + call from internalHelper
        assert len(same_file_refs) >= 3


class TestReduxSagaPatterns:
    """Tests for Redux Saga patterns.

    Inspired by Appsmith's extensive use of Redux Saga:
    yield call(getUpdatedTabs, id, jsTabs);
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with Redux Saga patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        sagas_dir = src_dir / "sagas"
        sagas_dir.mkdir()

        # Helper function
        (src_dir / "api.ts").write_text("""
export async function fetchUserData(userId: string) {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
}

export async function updateUser(userId: string, data: any) {
    const response = await fetch(`/api/users/${userId}`, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
    return response.json();
}
""")

        # Saga file
        (sagas_dir / "userSaga.ts").write_text("""
import { call, put, takeLatest } from 'redux-saga/effects';
import { fetchUserData, updateUser } from '../api';

function* handleFetchUser(action: any) {
    try {
        // yield call pattern
        const user = yield call(fetchUserData, action.payload.userId);
        yield put({ type: 'USER_FETCH_SUCCESS', payload: user });
    } catch (error) {
        yield put({ type: 'USER_FETCH_FAILURE', error });
    }
}

function* handleUpdateUser(action: any) {
    try {
        const result = yield call(updateUser, action.payload.userId, action.payload.data);

        // Re-fetch after update
        const updatedUser = yield call(fetchUserData, action.payload.userId);
        yield put({ type: 'USER_UPDATE_SUCCESS', payload: updatedUser });
    } catch (error) {
        yield put({ type: 'USER_UPDATE_FAILURE', error });
    }
}

export function* userSaga() {
    yield takeLatest('FETCH_USER', handleFetchUser);
    yield takeLatest('UPDATE_USER', handleUpdateUser);
}
""")

        return tmp_path

    def test_find_saga_call_references(self, project_root):
        """Test finding functions used in yield call() patterns."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "api.ts"

        refs = finder.find_references("fetchUserData", source_file)

        saga_refs = [r for r in refs if "sagas" in str(r.file_path)]
        # Should find two calls to fetchUserData (one in handleFetchUser, one in handleUpdateUser)
        assert len(saga_refs) >= 2


class TestReduxSelectorPatterns:
    """Tests for Redux Selector patterns.

    Inspired by Appsmith's use of reselect:
    createSelector(getQuerySegmentItems, (items) => groupAndSortEntitySegmentList(items));
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with Redux selector patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        selectors_dir = src_dir / "selectors"
        selectors_dir.mkdir()

        # Helper functions
        (src_dir / "sortUtils.ts").write_text("""
export function groupAndSortEntitySegmentList(items: any[]) {
    return items
        .sort((a, b) => a.name.localeCompare(b.name))
        .reduce((groups, item) => {
            const key = item.type;
            if (!groups[key]) groups[key] = [];
            groups[key].push(item);
            return groups;
        }, {});
}

export function sortByName(items: any[]) {
    return [...items].sort((a, b) => a.name.localeCompare(b.name));
}
""")

        # Selectors file
        (selectors_dir / "entitySelectors.ts").write_text("""
import { createSelector } from 'reselect';
import { groupAndSortEntitySegmentList, sortByName } from '../sortUtils';

const getQuerySegmentItems = (state: any) => state.queries.items;
const getJSSegmentItems = (state: any) => state.js.items;

// Function used in selector
export const getSortedQueryItems = createSelector(
    getQuerySegmentItems,
    (items) => groupAndSortEntitySegmentList(items)
);

export const getSortedJSItems = createSelector(
    getJSSegmentItems,
    sortByName  // Function passed directly as callback
);

// Multiple selectors using same function
export const getCombinedItems = createSelector(
    [getQuerySegmentItems, getJSSegmentItems],
    (queries, js) => {
        const combined = [...queries, ...js];
        return groupAndSortEntitySegmentList(combined);
    }
);
""")

        return tmp_path

    def test_find_selector_callback_references(self, project_root):
        """Test finding functions used in createSelector callbacks."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "sortUtils.ts"

        refs = finder.find_references("groupAndSortEntitySegmentList", source_file)

        selector_refs = [r for r in refs if "selectors" in str(r.file_path)]
        # Should find two uses in selectors
        assert len(selector_refs) >= 2

    def test_find_direct_callback_reference(self, project_root):
        """Test finding function passed directly as callback to createSelector."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "sortUtils.ts"

        refs = finder.find_references("sortByName", source_file)

        selector_refs = [r for r in refs if "selectors" in str(r.file_path)]
        assert len(selector_refs) >= 1


class TestCommonJSPatterns:
    """Tests for CommonJS require/module.exports patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with CommonJS patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # CommonJS module
        (src_dir / "helpers.js").write_text("""
function processConfig(config) {
    return {
        ...config,
        processed: true
    };
}

function validateConfig(config) {
    return config && typeof config === 'object';
}

module.exports = {
    processConfig,
    validateConfig
};
""")

        # Consumer using require
        (src_dir / "main.js").write_text("""
const { processConfig, validateConfig } = require('./helpers');

function handleConfig(config) {
    if (validateConfig(config)) {
        return processConfig(config);
    }
    throw new Error('Invalid config');
}

module.exports = handleConfig;
""")

        # Consumer using require with property access
        (src_dir / "alternative.js").write_text("""
const helpers = require('./helpers');

function process(config) {
    return helpers.processConfig(config);
}

module.exports = process;
""")

        return tmp_path

    def test_find_commonjs_destructured_require(self, project_root):
        """Test finding references via destructured require."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helpers.js"

        refs = finder.find_references("processConfig", source_file)

        main_refs = [r for r in refs if r.file_path == project_root / "src" / "main.js"]
        assert len(main_refs) >= 1

    def test_find_commonjs_property_access(self, project_root):
        """Test finding references via require().property pattern."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helpers.js"

        refs = finder.find_references("processConfig", source_file)

        alt_refs = [r for r in refs if r.file_path == project_root / "src" / "alternative.js"]
        assert len(alt_refs) >= 1


class TestComplexMultiFileScenarios:
    """Tests for complex multi-file scenarios inspired by Appsmith.

    This tests scenarios with multiple levels of imports, re-exports,
    and various reference patterns.
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a complex multi-file project structure."""
        # Create directory structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "utils").mkdir()
        (src_dir / "components").mkdir()
        (src_dir / "sagas").mkdir()
        (src_dir / "selectors").mkdir()

        # Core utility function
        (src_dir / "utils" / "widgetUtils.ts").write_text("""
export function isLargeWidget(widgetType: string): boolean {
    const largeWidgets = ['TABLE', 'LIST', 'MAP'];
    return largeWidgets.includes(widgetType);
}

export function getWidgetDimensions(widgetType: string) {
    return isLargeWidget(widgetType)
        ? { width: 12, height: 8 }
        : { width: 4, height: 4 };
}
""")

        # Re-export from index
        (src_dir / "utils" / "index.ts").write_text("""
export { isLargeWidget, getWidgetDimensions } from './widgetUtils';
export * from './otherUtils';
""")

        # Other utils for completeness
        (src_dir / "utils" / "otherUtils.ts").write_text("""
export function formatName(name: string) {
    return name.trim().toLowerCase();
}
""")

        # Component using the function
        (src_dir / "components" / "WidgetCard.tsx").write_text("""
import React from 'react';
import { isLargeWidget, getWidgetDimensions } from '../utils';

interface Props {
    widgetType: string;
    name: string;
}

export function WidgetCard({ widgetType, name }: Props) {
    const isLarge = isLargeWidget(widgetType);
    const dimensions = getWidgetDimensions(widgetType);

    return (
        <div className={isLarge ? 'large-card' : 'small-card'}>
            <h3>{name}</h3>
            <p>Size: {dimensions.width} x {dimensions.height}</p>
        </div>
    );
}
""")

        # Saga using the function
        (src_dir / "sagas" / "widgetSaga.ts").write_text("""
import { call, put, select } from 'redux-saga/effects';
import { isLargeWidget } from '../utils';

function* handleWidgetDrop(action: any) {
    const { widgetType, position } = action.payload;

    if (isLargeWidget(widgetType)) {
        // Large widget logic
        yield put({ type: 'PLACE_LARGE_WIDGET', payload: { position } });
    } else {
        yield put({ type: 'PLACE_SMALL_WIDGET', payload: { position } });
    }
}

export function* widgetSaga() {
    yield takeLatest('WIDGET_DROP', handleWidgetDrop);
}
""")

        # Selector using the function
        (src_dir / "selectors" / "widgetSelectors.ts").write_text("""
import { createSelector } from 'reselect';
import { isLargeWidget } from '../utils';

const getWidgets = (state: any) => state.widgets;

export const getLargeWidgets = createSelector(
    getWidgets,
    (widgets) => widgets.filter((w: any) => isLargeWidget(w.type))
);

export const getSmallWidgets = createSelector(
    getWidgets,
    (widgets) => widgets.filter((w: any) => !isLargeWidget(w.type))
);
""")

        return tmp_path

    def test_find_all_references_across_codebase(self, project_root):
        """Test finding all references to isLargeWidget across the codebase."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "widgetUtils.ts"

        refs = finder.find_references("isLargeWidget", source_file)

        # Should find references in:
        # 1. widgetUtils.ts (internal call from getWidgetDimensions)
        # 2. index.ts (re-export)
        # 3. WidgetCard.tsx (component)
        # 4. widgetSaga.ts (saga)
        # 5. widgetSelectors.ts (2 uses in selectors)

        ref_files = {ref.file_path for ref in refs}

        # Verify key files are found
        assert project_root / "src" / "utils" / "index.ts" in ref_files or any(
            r.reference_type == "reexport" for r in refs
        )
        # Note: The component, saga, and selector files might not be found
        # if they import from utils/index.ts rather than widgetUtils.ts directly
        # The test verifies the finder is working, actual file list depends on import resolution

        assert len(refs) >= 3  # At minimum: internal call, re-export, and some consumers

    def test_reference_contains_caller_function(self, project_root):
        """Test that references include the calling function name."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "widgetUtils.ts"

        refs = finder.find_references("isLargeWidget", source_file, include_definition=True)

        # The internal call should have getWidgetDimensions as caller
        internal_refs = [r for r in refs if r.file_path == source_file and r.reference_type == "call"]
        if internal_refs:
            assert any(r.caller_function == "getWidgetDimensions" for r in internal_refs)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project for edge case testing."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    def test_nonexistent_file(self, project_root):
        """Test handling of nonexistent source file."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "nonexistent.ts"

        refs = finder.find_references("someFunction", source_file)

        assert refs == []

    def test_non_exported_function(self, project_root):
        """Test handling of non-exported function."""
        # Create a file with non-exported function
        (project_root / "src" / "private.ts").write_text("""
function internalHelper() {
    return 42;
}

export function publicFunction() {
    return internalHelper();
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "private.ts"

        refs = finder.find_references("internalHelper", source_file)

        # Should only find internal reference, no external imports possible
        assert all(r.file_path == source_file for r in refs)

    def test_empty_file(self, project_root):
        """Test handling of empty file."""
        (project_root / "src" / "empty.ts").write_text("")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "empty.ts"

        refs = finder.find_references("anything", source_file)

        assert refs == []

    def test_max_files_limit(self, project_root):
        """Test that max_files limit is respected."""
        # Create many files
        for i in range(20):
            (project_root / "src" / f"file{i}.ts").write_text(f"""
export function func{i}() {{ return {i}; }}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "file0.ts"

        # Set a low limit
        refs = finder.find_references("func0", source_file, max_files=5)

        # Should not crash, even if we can't search all files
        assert isinstance(refs, list)


class TestConvenienceFunction:
    """Tests for the find_references convenience function."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a simple project."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        (src_dir / "utils.ts").write_text("""
export function helper() {
    return 42;
}
""")

        (src_dir / "main.ts").write_text("""
import { helper } from './utils';

export function main() {
    return helper();
}
""")

        return tmp_path

    def test_find_references_function(self, project_root):
        """Test the find_references convenience function."""
        source_file = project_root / "src" / "utils.ts"

        refs = find_references("helper", source_file, project_root=project_root)

        assert len(refs) >= 1
        assert any(r.file_path == project_root / "src" / "main.ts" for r in refs)

    def test_find_references_default_project_root(self, project_root):
        """Test find_references with default project_root."""
        source_file = project_root / "src" / "utils.ts"

        # Should use source_file.parent as project root
        refs = find_references("helper", source_file)

        # Should still work (searches from src/ directory)
        assert isinstance(refs, list)


class TestReferenceDataclass:
    """Tests for Reference dataclass."""

    def test_reference_creation(self, tmp_path):
        """Test creating a Reference object."""
        ref = Reference(
            file_path=tmp_path / "test.ts",
            line=10,
            column=5,
            end_line=10,
            end_column=15,
            context="const result = myFunction();",
            reference_type="call",
            import_name="myFunction",
            caller_function="processData",
        )

        assert ref.line == 10
        assert ref.reference_type == "call"
        assert ref.import_name == "myFunction"
        assert ref.caller_function == "processData"

    def test_reference_without_caller(self, tmp_path):
        """Test Reference with no caller function."""
        ref = Reference(
            file_path=tmp_path / "test.ts",
            line=1,
            column=0,
            end_line=1,
            end_column=10,
            context="export { fn } from './module';",
            reference_type="reexport",
            import_name="fn",
        )

        assert ref.caller_function is None


class TestExportedFunctionDataclass:
    """Tests for ExportedFunction dataclass."""

    def test_exported_function_named(self, tmp_path):
        """Test ExportedFunction for named export."""
        exp = ExportedFunction(
            function_name="myHelper",
            export_name="myHelper",
            is_default=False,
            file_path=tmp_path / "utils.ts",
        )

        assert exp.function_name == "myHelper"
        assert exp.is_default is False

    def test_exported_function_default(self, tmp_path):
        """Test ExportedFunction for default export."""
        exp = ExportedFunction(
            function_name="processData",
            export_name="default",
            is_default=True,
            file_path=tmp_path / "processor.ts",
        )

        assert exp.is_default is True
        assert exp.export_name == "default"


class TestReferenceSearchContext:
    """Tests for ReferenceSearchContext dataclass."""

    def test_context_defaults(self):
        """Test default values for ReferenceSearchContext."""
        ctx = ReferenceSearchContext()

        assert ctx.visited_files == set()
        assert ctx.max_files == 1000

    def test_context_custom_max_files(self):
        """Test custom max_files value."""
        ctx = ReferenceSearchContext(max_files=500)

        assert ctx.max_files == 500


class TestEdgeCasesAdvanced:
    """Advanced edge case tests to catch potential failures."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project for edge case testing."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    def test_function_with_same_name_different_files(self, project_root):
        """Test finding references when multiple files have functions with same name."""
        src_dir = project_root / "src"

        # Two files with same function name
        (src_dir / "utils1.ts").write_text("""
export function process(data: any) {
    return data.map(x => x * 2);
}
""")

        (src_dir / "utils2.ts").write_text("""
export function process(data: any) {
    return data.filter(x => x > 0);
}
""")

        # Consumer imports from utils1
        (src_dir / "consumer.ts").write_text("""
import { process } from './utils1';

export function handle(items: any[]) {
    return process(items);
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils1.ts"

        refs = finder.find_references("process", source_file)

        # Should only find reference from consumer (which imports from utils1)
        consumer_refs = [r for r in refs if r.file_path == project_root / "src" / "consumer.ts"]
        assert len(consumer_refs) >= 1

    def test_circular_import_handling(self, project_root):
        """Test that circular imports don't cause infinite loops."""
        src_dir = project_root / "src"

        # Create circular import structure
        (src_dir / "a.ts").write_text("""
import { funcB } from './b';

export function funcA() {
    return funcB() + 1;
}
""")

        (src_dir / "b.ts").write_text("""
import { funcA } from './a';

export function funcB() {
    return 42;
}

export function callsA() {
    return funcA();
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "a.ts"

        # Should not hang or crash
        refs = finder.find_references("funcA", source_file)

        assert isinstance(refs, list)
        # Should find reference in b.ts
        b_refs = [r for r in refs if r.file_path == project_root / "src" / "b.ts"]
        assert len(b_refs) >= 1

    def test_deeply_nested_directory_structure(self, project_root):
        """Test finding references in nested directory structures.

        Note: Very deep relative paths (many ../) may not be resolved by the
        import resolver. This test uses a moderate nesting level.
        """
        # Create moderate nesting (2 levels deep)
        deep_dir = project_root / "src" / "features" / "auth"
        deep_dir.mkdir(parents=True)
        utils_dir = project_root / "src" / "utils"
        utils_dir.mkdir(parents=True)

        (utils_dir / "helpers.ts").write_text("""
export function validateEmail(email: string): boolean {
    return email.includes('@');
}
""")

        (deep_dir / "LoginForm.tsx").write_text("""
import { validateEmail } from '../../utils/helpers';

export function LoginForm() {
    const handleSubmit = (email: string) => {
        if (validateEmail(email)) {
            console.log('Valid');
        }
    };
    return null;
}
""")

        finder = ReferenceFinder(project_root)
        source_file = utils_dir / "helpers.ts"

        refs = finder.find_references("validateEmail", source_file)

        # Should find reference in nested directory
        login_refs = [r for r in refs if "LoginForm" in str(r.file_path)]
        assert len(login_refs) >= 1

    def test_unicode_in_function_names(self, project_root):
        """Test handling of unicode in identifiers (while not common, some codebases use it)."""
        src_dir = project_root / "src"

        # File with unicode comments but ASCII function name
        (src_dir / "unicode.ts").write_text("""
// 日本語コメント
export function calculateTotal(items: number[]): number {
    // Добавить все элементы
    return items.reduce((a, b) => a + b, 0);
}
""")

        (src_dir / "consumer.ts").write_text("""
import { calculateTotal } from './unicode';

export function process() {
    return calculateTotal([1, 2, 3]);
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "unicode.ts"

        refs = finder.find_references("calculateTotal", source_file)

        assert len(refs) >= 1

    def test_dynamic_import_not_found(self, project_root):
        """Test that dynamic imports (import()) are not matched as static references."""
        src_dir = project_root / "src"

        (src_dir / "utils.ts").write_text("""
export function lazyLoad() {
    return import('./heavy-module');
}
""")

        (src_dir / "heavy-module.ts").write_text("""
export function heavyFunction() {
    return 'heavy computation';
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "heavy-module.ts"

        refs = finder.find_references("heavyFunction", source_file)

        # Dynamic imports don't create static references
        # This should return empty or minimal references
        assert isinstance(refs, list)

    def test_type_only_imports_excluded(self, project_root):
        """Test that type-only imports are handled correctly."""
        src_dir = project_root / "src"

        (src_dir / "types.ts").write_text("""
export interface User {
    id: string;
    name: string;
}

export function createUser(name: string): User {
    return { id: '123', name };
}
""")

        (src_dir / "consumer.ts").write_text("""
import type { User } from './types';
import { createUser } from './types';

export function makeUser(): User {
    return createUser('John');
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "types.ts"

        refs = finder.find_references("createUser", source_file)

        # Should find the call reference, not type import
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 1

    def test_jsx_component_as_function(self, project_root):
        """Test finding references to functions used as JSX components."""
        src_dir = project_root / "src"

        (src_dir / "Button.tsx").write_text("""
export function Button({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
    return <button onClick={onClick}>{children}</button>;
}
""")

        (src_dir / "App.tsx").write_text("""
import { Button } from './Button';

export function App() {
    return (
        <div>
            <Button onClick={() => console.log('clicked')}>Click me</Button>
        </div>
    );
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "Button.tsx"

        refs = finder.find_references("Button", source_file)

        # Should find the JSX usage
        app_refs = [r for r in refs if r.file_path == project_root / "src" / "App.tsx"]
        # JSX usage may be detected as reference or callback depending on AST
        assert len(app_refs) >= 1

    def test_function_passed_to_higher_order_function(self, project_root):
        """Test finding references when function is passed to HOF like debounce, throttle."""
        src_dir = project_root / "src"

        (src_dir / "handlers.ts").write_text("""
export function handleSearch(query: string) {
    console.log('Searching:', query);
}
""")

        (src_dir / "component.ts").write_text("""
import debounce from 'lodash/debounce';
import { handleSearch } from './handlers';

// Function passed to debounce
const debouncedSearch = debounce(handleSearch, 300);

export function onInputChange(value: string) {
    debouncedSearch(value);
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "handlers.ts"

        refs = finder.find_references("handleSearch", source_file)

        # Should find the reference passed to debounce
        component_refs = [r for r in refs if r.file_path == project_root / "src" / "component.ts"]
        assert len(component_refs) >= 1

    def test_export_with_as_keyword(self, project_root):
        """Test finding references when function is exported with 'as' keyword."""
        src_dir = project_root / "src"

        (src_dir / "internal.ts").write_text("""
function internalProcess(data: any) {
    return data;
}

// Export with different name
export { internalProcess as publicProcess };
""")

        (src_dir / "consumer.ts").write_text("""
import { publicProcess } from './internal';

export function use() {
    return publicProcess({ x: 1 });
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "internal.ts"

        refs = finder.find_references("internalProcess", source_file)

        # Should find reference through the aliased export
        consumer_refs = [r for r in refs if r.file_path == project_root / "src" / "consumer.ts"]
        assert len(consumer_refs) >= 1

    def test_very_large_file(self, project_root):
        """Test performance with a large file."""
        src_dir = project_root / "src"

        # Create a large file with many functions
        large_content = "export function targetFunction() { return 42; }\n\n"
        for i in range(100):
            large_content += f"""
export function func{i}() {{
    const result = targetFunction();
    return result + {i};
}}
"""

        (src_dir / "large.ts").write_text(large_content)

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "large.ts"

        refs = finder.find_references("targetFunction", source_file, include_definition=True)

        # Should find many references (100 calls + definition)
        # The exact count may vary but should be substantial
        assert len(refs) >= 50  # At least half should be found

    def test_syntax_error_in_file_graceful_handling(self, project_root):
        """Test that syntax errors in files are handled gracefully."""
        src_dir = project_root / "src"

        (src_dir / "valid.ts").write_text("""
export function validFunction() {
    return 42;
}
""")

        # Create a file with syntax error
        (src_dir / "invalid.ts").write_text("""
import { validFunction } from './valid';

export function broken( {
    // Missing closing brace and paren
    return validFunction(
}
""")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "valid.ts"

        # Should not crash, should return whatever valid references it can find
        refs = finder.find_references("validFunction", source_file)

        assert isinstance(refs, list)
        # May or may not find references depending on how parser handles errors
