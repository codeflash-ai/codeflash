"""Tests for JavaScript/TypeScript code context extraction.

This module tests the extract_code_context method and related functionality
for JavaScript and TypeScript, mirroring the Python tests in test_code_context_extractor.py.

The tests cover:
- Simple functions and their dependencies
- Class methods with helpers and sibling method calls
- Helper functions in the same file
- Helper functions from imported files (cross-file)
- Global variables and constants
- Type definitions (TypeScript)
- JSDoc comments
- Constructor and fields context
- Nested dependencies (helper of helper)
- Circular dependencies

All assertions use strict string equality to verify exact extraction output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.context.code_context_extractor import get_code_optimization_context_for_language
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport


@pytest.fixture
def js_support():
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


@pytest.fixture
def ts_support():
    """Create a TypeScriptSupport instance."""
    return TypeScriptSupport()


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory structure."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    return project_root


class TestSimpleFunctionContext:
    """Tests for simple function code context extraction with strict assertions."""

    def test_simple_function_no_dependencies(self, js_support, temp_project):
        """Test extracting context for a simple standalone function without any dependencies."""
        code = """\
function add(a, b) {
    return a + b;
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        assert len(functions) == 1
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function add(a, b) {
    return a + b;
}
"""
        assert context.target_code == expected_target_code
        assert context.language == Language.JAVASCRIPT
        assert context.target_file == file_path
        assert context.helper_functions == []
        assert context.read_only_context == ""
        assert context.imports == []

    def test_arrow_function_with_implicit_return(self, js_support, temp_project):
        """Test extracting an arrow function with implicit return."""
        code = """\
const multiply = (a, b) => a * b;
"""
        file_path = temp_project / "math.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        assert len(functions) == 1
        func = functions[0]
        assert func.function_name == "multiply"

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
const multiply = (a, b) => a * b;
"""
        assert context.target_code == expected_target_code
        assert context.helper_functions == []
        assert context.read_only_context == ""


class TestJSDocExtraction:
    """Tests for JSDoc comment extraction with complex documentation patterns."""

    def test_function_with_simple_jsdoc(self, js_support, temp_project):
        """Test extracting function with simple JSDoc - exact string match."""
        code = """\
/**
 * Adds two numbers together.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
/**
 * Adds two numbers together.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}
"""
        assert context.target_code == expected_target_code
        assert context.helper_functions == []

    def test_function_with_complex_jsdoc_types(self, js_support, temp_project):
        """Test JSDoc with complex type annotations including generics, unions, and callbacks."""
        code = """\
/**
 * Processes an array of items with a callback function.
 *
 * This function iterates over each item and applies the transformation.
 *
 * @template T - The type of items in the input array
 * @template U - The type of items in the output array
 * @param {Array<T>} items - The input array to process
 * @param {function(T, number): U} callback - Transformation function
 * @param {Object} [options] - Optional configuration
 * @param {boolean} [options.parallel=false] - Whether to process in parallel
 * @param {number} [options.chunkSize=100] - Size of processing chunks
 * @returns {Promise<Array<U>>} The transformed array
 * @throws {TypeError} If items is not an array
 * @example
 * const doubled = await processItems([1, 2, 3], x => x * 2);
 * // returns [2, 4, 6]
 */
async function processItems(items, callback, options = {}) {
    const { parallel = false, chunkSize = 100 } = options;

    if (!Array.isArray(items)) {
        throw new TypeError('items must be an array');
    }

    const results = [];
    for (let i = 0; i < items.length; i++) {
        results.push(callback(items[i], i));
    }

    return results;
}
"""
        file_path = temp_project / "processor.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
/**
 * Processes an array of items with a callback function.
 *
 * This function iterates over each item and applies the transformation.
 *
 * @template T - The type of items in the input array
 * @template U - The type of items in the output array
 * @param {Array<T>} items - The input array to process
 * @param {function(T, number): U} callback - Transformation function
 * @param {Object} [options] - Optional configuration
 * @param {boolean} [options.parallel=false] - Whether to process in parallel
 * @param {number} [options.chunkSize=100] - Size of processing chunks
 * @returns {Promise<Array<U>>} The transformed array
 * @throws {TypeError} If items is not an array
 * @example
 * const doubled = await processItems([1, 2, 3], x => x * 2);
 * // returns [2, 4, 6]
 */
async function processItems(items, callback, options = {}) {
    const { parallel = false, chunkSize = 100 } = options;

    if (!Array.isArray(items)) {
        throw new TypeError('items must be an array');
    }

    const results = [];
    for (let i = 0; i < items.length; i++) {
        results.push(callback(items[i], i));
    }

    return results;
}
"""
        assert context.target_code == expected_target_code

    def test_class_with_jsdoc_on_class_and_methods(self, js_support, temp_project):
        """Test class where both the class and method have JSDoc comments."""
        code = """\
/**
 * A cache implementation with TTL support.
 *
 * @class CacheManager
 * @description Provides in-memory caching with automatic expiration.
 */
class CacheManager {
    /**
     * Creates a new cache manager.
     * @param {number} defaultTTL - Default time-to-live in milliseconds
     */
    constructor(defaultTTL = 60000) {
        this.cache = new Map();
        this.defaultTTL = defaultTTL;
    }

    /**
     * Retrieves a value from cache or computes it.
     *
     * If the key exists and hasn't expired, returns the cached value.
     * Otherwise, calls the factory function and caches the result.
     *
     * @param {string} key - The cache key
     * @param {function(): T} factory - Factory function to compute value
     * @param {number} [ttl] - Optional TTL override
     * @returns {T} The cached or computed value
     * @template T
     */
    getOrCompute(key, factory, ttl) {
        const existing = this.cache.get(key);
        if (existing && existing.expiry > Date.now()) {
            return existing.value;
        }

        const value = factory();
        const expiry = Date.now() + (ttl || this.defaultTTL);
        this.cache.set(key, { value, expiry });
        return value;
    }
}
"""
        file_path = temp_project / "cache.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        get_or_compute = next(f for f in functions if f.function_name == "getOrCompute")

        context = js_support.extract_code_context(get_or_compute, temp_project, temp_project)

        expected_target_code = """\
/**
 * A cache implementation with TTL support.
 *
 * @class CacheManager
 * @description Provides in-memory caching with automatic expiration.
 */
class CacheManager {
    /**
     * Creates a new cache manager.
     * @param {number} defaultTTL - Default time-to-live in milliseconds
     */
    constructor(defaultTTL = 60000) {
        this.cache = new Map();
        this.defaultTTL = defaultTTL;
    }

    /**
     * Retrieves a value from cache or computes it.
     *
     * If the key exists and hasn't expired, returns the cached value.
     * Otherwise, calls the factory function and caches the result.
     *
     * @param {string} key - The cache key
     * @param {function(): T} factory - Factory function to compute value
     * @param {number} [ttl] - Optional TTL override
     * @returns {T} The cached or computed value
     * @template T
     */
    getOrCompute(key, factory, ttl) {
        const existing = this.cache.get(key);
        if (existing && existing.expiry > Date.now()) {
            return existing.value;
        }

        const value = factory();
        const expiry = Date.now() + (ttl || this.defaultTTL);
        this.cache.set(key, { value, expiry });
        return value;
    }
}
"""
        assert context.target_code == expected_target_code
        assert js_support.validate_syntax(context.target_code) is True

    def test_jsdoc_with_typedef_and_callback(self, js_support, temp_project):
        """Test JSDoc with @typedef and @callback definitions referenced in function."""
        code = """\
/**
 * @typedef {Object} ValidationResult
 * @property {boolean} valid - Whether validation passed
 * @property {string[]} errors - List of error messages
 * @property {Object.<string, string>} fieldErrors - Field-specific errors
 */

/**
 * @callback ValidatorFunction
 * @param {*} value - The value to validate
 * @param {Object} context - Validation context
 * @returns {ValidationResult}
 */

const EMAIL_REGEX = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;

/**
 * Validates user input data.
 * @param {Object} data - The data to validate
 * @param {ValidatorFunction[]} validators - Array of validator functions
 * @returns {ValidationResult} Combined validation result
 */
function validateUserData(data, validators) {
    const errors = [];
    const fieldErrors = {};

    for (const validator of validators) {
        const result = validator(data, { strict: true });
        if (!result.valid) {
            errors.push(...result.errors);
            Object.assign(fieldErrors, result.fieldErrors);
        }
    }

    if (data.email && !EMAIL_REGEX.test(data.email)) {
        errors.push('Invalid email format');
        fieldErrors.email = 'Invalid email format';
    }

    return {
        valid: errors.length === 0,
        errors,
        fieldErrors
    };
}
"""
        file_path = temp_project / "validator.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = next(f for f in functions if f.function_name == "validateUserData")

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
/**
 * Validates user input data.
 * @param {Object} data - The data to validate
 * @param {ValidatorFunction[]} validators - Array of validator functions
 * @returns {ValidationResult} Combined validation result
 */
function validateUserData(data, validators) {
    const errors = [];
    const fieldErrors = {};

    for (const validator of validators) {
        const result = validator(data, { strict: true });
        if (!result.valid) {
            errors.push(...result.errors);
            Object.assign(fieldErrors, result.fieldErrors);
        }
    }

    if (data.email && !EMAIL_REGEX.test(data.email)) {
        errors.push('Invalid email format');
        fieldErrors.email = 'Invalid email format';
    }

    return {
        valid: errors.length === 0,
        errors,
        fieldErrors
    };
}
"""
        assert context.target_code == expected_target_code
        # EMAIL_REGEX should be in read_only_context - exact match
        expected_read_only = "const EMAIL_REGEX = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;"
        assert context.read_only_context == expected_read_only


class TestGlobalVariablesAndConstants:
    """Tests for global variables and constants extraction with strict assertions."""

    def test_function_with_multiple_complex_constants(self, js_support, temp_project):
        """Test function using multiple global constants of different types."""
        code = """\
const API_BASE_URL = 'https://api.example.com/v2';
const DEFAULT_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];
const HTTP_STATUS = {
    OK: 200,
    CREATED: 201,
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    NOT_FOUND: 404,
    SERVER_ERROR: 500
};
const UNUSED_CONFIG = { debug: false };

async function fetchWithRetry(endpoint, options = {}) {
    const url = API_BASE_URL + endpoint;
    let lastError;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                ...options,
                timeout: DEFAULT_TIMEOUT
            });

            if (response.status === HTTP_STATUS.OK) {
                return response.json();
            }

            if (response.status >= HTTP_STATUS.SERVER_ERROR) {
                throw new Error('Server error');
            }

            return null;
        } catch (error) {
            lastError = error;
            if (attempt < MAX_RETRIES - 1) {
                await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt]));
            }
        }
    }

    throw lastError;
}
"""
        file_path = temp_project / "api.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = next(f for f in functions if f.function_name == "fetchWithRetry")

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
async function fetchWithRetry(endpoint, options = {}) {
    const url = API_BASE_URL + endpoint;
    let lastError;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                ...options,
                timeout: DEFAULT_TIMEOUT
            });

            if (response.status === HTTP_STATUS.OK) {
                return response.json();
            }

            if (response.status >= HTTP_STATUS.SERVER_ERROR) {
                throw new Error('Server error');
            }

            return null;
        } catch (error) {
            lastError = error;
            if (attempt < MAX_RETRIES - 1) {
                await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt]));
            }
        }
    }

    throw lastError;
}
"""
        assert context.target_code == expected_target_code

        # All used constants should be in read_only_context - exact match
        expected_read_only = """\
const API_BASE_URL = 'https://api.example.com/v2';
const DEFAULT_TIMEOUT = 30000;
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];
const HTTP_STATUS = {
    OK: 200,
    CREATED: 201,
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    NOT_FOUND: 404,
    SERVER_ERROR: 500
};"""
        assert context.read_only_context == expected_read_only

    def test_function_with_regex_and_template_constants(self, js_support, temp_project):
        """Test function with regex patterns and template literal constants."""
        code = """\
const PATTERNS = {
    email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/,
    phone: /^\\+?[1-9]\\d{1,14}$/,
    url: /^https?:\\/\\/[^\\s/$.?#].[^\\s]*$/i
};

const ERROR_MESSAGES = {
    email: 'Please enter a valid email address',
    phone: 'Please enter a valid phone number',
    url: 'Please enter a valid URL'
};

function validateField(value, fieldType) {
    const pattern = PATTERNS[fieldType];
    if (!pattern) {
        return { valid: true, error: null };
    }

    const valid = pattern.test(value);
    return {
        valid,
        error: valid ? null : ERROR_MESSAGES[fieldType]
    };
}
"""
        file_path = temp_project / "validation.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function validateField(value, fieldType) {
    const pattern = PATTERNS[fieldType];
    if (!pattern) {
        return { valid: true, error: null };
    }

    const valid = pattern.test(value);
    return {
        valid,
        error: valid ? null : ERROR_MESSAGES[fieldType]
    };
}
"""
        assert context.target_code == expected_target_code

        # Exact match for read_only_context (globals joined with single newline)
        expected_read_only = """\
const PATTERNS = {
    email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/,
    phone: /^\\+?[1-9]\\d{1,14}$/,
    url: /^https?:\\/\\/[^\\s/$.?#].[^\\s]*$/i
};
const ERROR_MESSAGES = {
    email: 'Please enter a valid email address',
    phone: 'Please enter a valid phone number',
    url: 'Please enter a valid URL'
};"""
        assert context.read_only_context == expected_read_only


class TestSameFileHelperFunctions:
    """Tests for helper functions discovery within the same file."""

    def test_function_with_chain_of_helpers(self, js_support, temp_project):
        """Test function calling helper that calls another helper (transitive dependencies)."""
        code = """\
function sanitizeString(str) {
    return str.trim().toLowerCase();
}

function normalizeInput(input) {
    const sanitized = sanitizeString(input);
    return sanitized.replace(/\\s+/g, '-');
}

function processUserInput(rawInput) {
    const normalized = normalizeInput(rawInput);
    return {
        original: rawInput,
        processed: normalized,
        length: normalized.length
    };
}
"""
        file_path = temp_project / "processor.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        process_func = next(f for f in functions if f.function_name == "processUserInput")

        context = js_support.extract_code_context(process_func, temp_project, temp_project)

        expected_target_code = """\
function processUserInput(rawInput) {
    const normalized = normalizeInput(rawInput);
    return {
        original: rawInput,
        processed: normalized,
        length: normalized.length
    };
}
"""
        assert context.target_code == expected_target_code

        # Direct helper normalizeInput should be found - exact list match
        helper_names = [h.name for h in context.helper_functions]
        assert helper_names == ["normalizeInput"]

    def test_function_with_multiple_unrelated_helpers(self, js_support, temp_project):
        """Test function calling multiple independent helper functions."""
        code = """\
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function formatCurrency(amount) {
    return '$' + amount.toFixed(2);
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

function unusedFormatter() {
    return 'not used';
}

function generateReport(data) {
    const date = formatDate(new Date(data.timestamp));
    const revenue = formatCurrency(data.revenue);
    const growth = formatPercentage(data.growth);

    return {
        reportDate: date,
        totalRevenue: revenue,
        growthRate: growth
    };
}
"""
        file_path = temp_project / "report.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        report_func = next(f for f in functions if f.function_name == "generateReport")

        context = js_support.extract_code_context(report_func, temp_project, temp_project)

        expected_target_code = """\
function generateReport(data) {
    const date = formatDate(new Date(data.timestamp));
    const revenue = formatCurrency(data.revenue);
    const growth = formatPercentage(data.growth);

    return {
        reportDate: date,
        totalRevenue: revenue,
        growthRate: growth
    };
}
"""
        assert context.target_code == expected_target_code

        # All three used helpers should be found
        helper_names = sorted([h.name for h in context.helper_functions])
        assert helper_names == ["formatCurrency", "formatDate", "formatPercentage"]

        # Verify helper source code exactly
        for helper in context.helper_functions:
            if helper.name == "formatDate":
                expected = """\
function formatDate(date) {
    return date.toISOString().split('T')[0];
}
"""
                assert helper.source_code == expected
            elif helper.name == "formatCurrency":
                expected = """\
function formatCurrency(amount) {
    return '$' + amount.toFixed(2);
}
"""
                assert helper.source_code == expected
            elif helper.name == "formatPercentage":
                expected = """\
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}
"""
                assert helper.source_code == expected


class TestClassMethodWithSiblingMethods:
    """Tests for class methods calling other methods in the same class."""

    def test_graph_topological_sort(self, js_support, temp_project):
        """Test graph class with topological sort - similar to Python test_class_method_dependencies."""
        code = """\
class Graph {
    constructor(vertices) {
        this.graph = new Map();
        this.V = vertices;
    }

    addEdge(u, v) {
        if (!this.graph.has(u)) {
            this.graph.set(u, []);
        }
        this.graph.get(u).push(v);
    }

    topologicalSortUtil(v, visited, stack) {
        visited[v] = true;

        const neighbors = this.graph.get(v) || [];
        for (const i of neighbors) {
            if (visited[i] === false) {
                this.topologicalSortUtil(i, visited, stack);
            }
        }

        stack.unshift(v);
    }

    topologicalSort() {
        const visited = new Array(this.V).fill(false);
        const stack = [];

        for (let i = 0; i < this.V; i++) {
            if (visited[i] === false) {
                this.topologicalSortUtil(i, visited, stack);
            }
        }

        return stack;
    }
}
"""
        file_path = temp_project / "graph.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        topo_sort = next(f for f in functions if f.function_name == "topologicalSort")

        context = js_support.extract_code_context(topo_sort, temp_project, temp_project)

        # The extracted code should include class wrapper with constructor
        expected_target_code = """\
class Graph {
    constructor(vertices) {
        this.graph = new Map();
        this.V = vertices;
    }

    topologicalSort() {
        const visited = new Array(this.V).fill(false);
        const stack = [];

        for (let i = 0; i < this.V; i++) {
            if (visited[i] === false) {
                this.topologicalSortUtil(i, visited, stack);
            }
        }

        return stack;
    }
}
"""
        assert context.target_code == expected_target_code
        assert js_support.validate_syntax(context.target_code) is True

    def test_class_method_using_nested_helper_class(self, js_support, temp_project):
        """Test class method that uses another class as a helper - mirrors Python HelperClass test."""
        code = """\
class HelperClass {
    constructor(name) {
        this.name = name;
    }

    innocentBystander() {
        return 'not used';
    }

    helperMethod() {
        return this.name;
    }
}

class NestedHelper {
    constructor(name) {
        this.name = name;
    }

    nestedMethod() {
        return this.name;
    }
}

function mainMethod() {
    return 'hello';
}

class MainClass {
    constructor(name) {
        this.name = name;
    }

    mainMethod() {
        this.name = new NestedHelper('test').nestedMethod();
        return new HelperClass(this.name).helperMethod();
    }
}
"""
        file_path = temp_project / "classes.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        main_method = next(f for f in functions if f.function_name == "mainMethod" and f.class_name == "MainClass")

        context = js_support.extract_code_context(main_method, temp_project, temp_project)

        expected_target_code = """\
class MainClass {
    constructor(name) {
        this.name = name;
    }

    mainMethod() {
        this.name = new NestedHelper('test').nestedMethod();
        return new HelperClass(this.name).helperMethod();
    }
}
"""
        assert context.target_code == expected_target_code
        assert js_support.validate_syntax(context.target_code) is True


class TestMultiFileHelperExtraction:
    """Tests for helper functions extracted from imported files."""

    def test_helper_from_another_file_commonjs(self, js_support, temp_project):
        """Test function importing helper from another file via CommonJS - mirrors Python bubble_sort_helper."""
        # Create helper file with its own import
        helper_code = """\
const mathUtils = require('./math_utils');

function sorter(arr) {
    arr.sort();
    const x = mathUtils.sqrt(2);
    console.log(x);
    return arr;
}

module.exports = { sorter };
"""
        helper_path = temp_project / "bubble_sort_with_math.js"
        helper_path.write_text(helper_code, encoding="utf-8")

        # Create main file that imports the helper
        main_code = """\
const { sorter } = require('./bubble_sort_with_math');

function sortFromAnotherFile(arr) {
    const sortedArr = sorter(arr);
    return sortedArr;
}

module.exports = { sortFromAnotherFile };
"""
        main_path = temp_project / "bubble_sort_imported.js"
        main_path.write_text(main_code, encoding="utf-8")

        functions = js_support.discover_functions(main_path)
        main_func = next(f for f in functions if f.function_name == "sortFromAnotherFile")

        context = js_support.extract_code_context(main_func, temp_project, temp_project)

        expected_target_code = """\
function sortFromAnotherFile(arr) {
    const sortedArr = sorter(arr);
    return sortedArr;
}
"""
        assert context.target_code == expected_target_code

        # Import should be captured - exact match
        assert context.imports == ["const { sorter } = require('./bubble_sort_with_math');"]

    def test_helper_from_another_file_esm(self, js_support, temp_project):
        """Test ES module imports with named and default exports."""
        # Create utility module with multiple exports
        utils_code = """\
export function double(x) {
    return x * 2;
}

export function triple(x) {
    return x * 3;
}

export function square(x) {
    return x * x;
}

export default function identity(x) {
    return x;
}
"""
        utils_path = temp_project / "utils.js"
        utils_path.write_text(utils_code, encoding="utf-8")

        # Create main module with selective imports
        main_code = """\
import identity, { double, triple } from './utils';

function processNumber(n) {
    const base = identity(n);
    return double(base) + triple(base);
}

export { processNumber };
"""
        main_path = temp_project / "main.js"
        main_path.write_text(main_code, encoding="utf-8")

        functions = js_support.discover_functions(main_path)
        process_func = next(f for f in functions if f.function_name == "processNumber")

        context = js_support.extract_code_context(process_func, temp_project, temp_project)

        expected_target_code = """\
function processNumber(n) {
    const base = identity(n);
    return double(base) + triple(base);
}
"""
        assert context.target_code == expected_target_code

        # Import should be captured - exact match
        assert context.imports == ["import identity, { double, triple } from './utils';"]

    def test_chained_imports_across_three_files(self, js_support, temp_project):
        """Test helper chain: main -> middleware -> core."""
        # Create core utility
        core_code = """\
export function validateInput(input) {
    return input !== null && input !== undefined;
}

export function sanitizeInput(input) {
    return String(input).trim();
}
"""
        core_path = temp_project / "core.js"
        core_path.write_text(core_code, encoding="utf-8")

        # Create middleware that uses core
        middleware_code = """\
import { validateInput, sanitizeInput } from './core';

export function processInput(input) {
    if (!validateInput(input)) {
        throw new Error('Invalid input');
    }
    return sanitizeInput(input);
}

export function transformInput(input) {
    const processed = processInput(input);
    return processed.toUpperCase();
}
"""
        middleware_path = temp_project / "middleware.js"
        middleware_path.write_text(middleware_code, encoding="utf-8")

        # Create main that uses middleware
        main_code = """\
import { transformInput } from './middleware';

function handleUserInput(rawInput) {
    try {
        const result = transformInput(rawInput);
        return { success: true, data: result };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

export { handleUserInput };
"""
        main_path = temp_project / "main.js"
        main_path.write_text(main_code, encoding="utf-8")

        functions = js_support.discover_functions(main_path)
        handle_func = next(f for f in functions if f.function_name == "handleUserInput")

        context = js_support.extract_code_context(handle_func, temp_project, temp_project)

        expected_target_code = """\
function handleUserInput(rawInput) {
    try {
        const result = transformInput(rawInput);
        return { success: true, data: result };
    } catch (error) {
        return { success: false, error: error.message };
    }
}
"""
        assert context.target_code == expected_target_code

        # Import should be captured - exact match
        assert context.imports == ["import { transformInput } from './middleware';"]


class TestTypeScriptSpecificContext:
    """Tests for TypeScript-specific code context extraction."""

    def test_function_with_complex_generic_types(self, ts_support, temp_project):
        """Test TypeScript function with complex generic constraints and types."""
        code = """\
interface Identifiable {
    id: string;
}

interface Timestamped {
    createdAt: Date;
    updatedAt: Date;
}

type Entity<T> = T & Identifiable & Timestamped;

function createEntity<T extends object>(data: T): Entity<T> {
    const now = new Date();
    return {
        ...data,
        id: Math.random().toString(36).substring(2),
        createdAt: now,
        updatedAt: now
    };
}
"""
        file_path = temp_project / "entity.ts"
        file_path.write_text(code, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        func = functions[0]

        context = ts_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function createEntity<T extends object>(data: T): Entity<T> {
    const now = new Date();
    return {
        ...data,
        id: Math.random().toString(36).substring(2),
        createdAt: now,
        updatedAt: now
    };
}
"""
        assert context.target_code == expected_target_code

        # Type definitions should be in read_only_context - exact match
        expected_read_only = """\
interface Identifiable {
    id: string;
}

interface Timestamped {
    createdAt: Date;
    updatedAt: Date;
}

type Entity<T> = T & Identifiable & Timestamped;"""
        assert context.read_only_context == expected_read_only

    def test_class_with_private_fields_and_typed_methods(self, ts_support, temp_project):
        """Test TypeScript class with private fields, readonly properties, and typed methods."""
        code = """\
interface CacheEntry<T> {
    value: T;
    expiry: number;
}

interface CacheConfig {
    defaultTTL: number;
    maxSize: number;
}

class TypedCache<T> {
    private readonly cache: Map<string, CacheEntry<T>>;
    private readonly config: CacheConfig;

    constructor(config: Partial<CacheConfig> = {}) {
        this.config = {
            defaultTTL: config.defaultTTL ?? 60000,
            maxSize: config.maxSize ?? 1000
        };
        this.cache = new Map();
    }

    get(key: string): T | undefined {
        const entry = this.cache.get(key);
        if (!entry) {
            return undefined;
        }
        if (entry.expiry < Date.now()) {
            this.cache.delete(key);
            return undefined;
        }
        return entry.value;
    }

    set(key: string, value: T, ttl?: number): void {
        if (this.cache.size >= this.config.maxSize) {
            this.evictOldest();
        }
        this.cache.set(key, {
            value,
            expiry: Date.now() + (ttl ?? this.config.defaultTTL)
        });
    }

    private evictOldest(): void {
        const firstKey = this.cache.keys().next().value;
        if (firstKey) {
            this.cache.delete(firstKey);
        }
    }
}
"""
        file_path = temp_project / "cache.ts"
        file_path.write_text(code, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        get_method = next(f for f in functions if f.function_name == "get")

        context = ts_support.extract_code_context(get_method, temp_project, temp_project)

        expected_target_code = """\
class TypedCache {
    private readonly cache: Map<string, CacheEntry<T>>;
    private readonly config: CacheConfig;

    constructor(config: Partial<CacheConfig> = {}) {
        this.config = {
            defaultTTL: config.defaultTTL ?? 60000,
            maxSize: config.maxSize ?? 1000
        };
        this.cache = new Map();
    }

    get(key: string): T | undefined {
        const entry = this.cache.get(key);
        if (!entry) {
            return undefined;
        }
        if (entry.expiry < Date.now()) {
            this.cache.delete(key);
            return undefined;
        }
        return entry.value;
    }
}
"""
        assert context.target_code == expected_target_code
        assert ts_support.validate_syntax(context.target_code) is True

        # Interfaces should be in read_only_context - exact match
        expected_read_only = """\
interface CacheEntry<T> {
    value: T;
    expiry: number;
}

interface CacheConfig {
    defaultTTL: number;
    maxSize: number;
}"""
        assert context.read_only_context == expected_read_only

    def test_typescript_with_type_imports(self, ts_support, temp_project):
        """Test TypeScript with type-only imports."""
        # Create types file
        types_code = """\
export interface User {
    id: string;
    name: string;
    email: string;
}

export interface CreateUserInput {
    name: string;
    email: string;
}

export type UserRole = 'admin' | 'user' | 'guest';
"""
        types_path = temp_project / "types.ts"
        types_path.write_text(types_code, encoding="utf-8")

        # Create service file that imports types
        service_code = """\
import type { User, CreateUserInput, UserRole } from './types';

const DEFAULT_ROLE: UserRole = 'user';

function createUser(input: CreateUserInput, role: UserRole = DEFAULT_ROLE): User {
    return {
        id: Math.random().toString(36).substring(2),
        name: input.name,
        email: input.email
    };
}

export { createUser };
"""
        service_path = temp_project / "service.ts"
        service_path.write_text(service_code, encoding="utf-8")

        functions = ts_support.discover_functions(service_path)
        func = next(f for f in functions if f.function_name == "createUser")

        context = ts_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function createUser(input: CreateUserInput, role: UserRole = DEFAULT_ROLE): User {
    return {
        id: Math.random().toString(36).substring(2),
        name: input.name,
        email: input.email
    };
}
"""
        assert context.target_code == expected_target_code

        # read_only_context should include imported type definitions and local constants
        expected_read_only = """\
// From types.ts

interface User {
    id: string;
    name: string;
    email: string;
}

interface CreateUserInput {
    name: string;
    email: string;
}

type UserRole = 'admin' | 'user' | 'guest';

const DEFAULT_ROLE: UserRole = 'user';"""
        assert context.read_only_context == expected_read_only

        # Import should be captured - exact match
        assert context.imports == ["import type { User, CreateUserInput, UserRole } from './types';"]


class TestRecursionAndCircularDependencies:
    """Tests for handling recursive functions and circular dependencies."""

    def test_self_recursive_factorial(self, js_support, temp_project):
        """Test self-recursive function does not list itself as helper."""
        code = """\
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""
        assert context.target_code == expected_target_code
        assert context.helper_functions == []

    def test_mutually_recursive_even_odd(self, js_support, temp_project):
        """Test mutually recursive functions."""
        code = """\
function isEven(n) {
    if (n === 0) return true;
    return isOdd(n - 1);
}

function isOdd(n) {
    if (n === 0) return false;
    return isEven(n - 1);
}
"""
        file_path = temp_project / "parity.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        is_even = next(f for f in functions if f.function_name == "isEven")

        context = js_support.extract_code_context(is_even, temp_project, temp_project)

        expected_target_code = """\
function isEven(n) {
    if (n === 0) return true;
    return isOdd(n - 1);
}
"""
        assert context.target_code == expected_target_code

        # isOdd should be a helper
        helper_names = [h.name for h in context.helper_functions]
        assert helper_names == ["isOdd"]

        # Verify helper source
        assert context.helper_functions[0].source_code == """\
function isOdd(n) {
    if (n === 0) return false;
    return isEven(n - 1);
}
"""

    def test_complex_recursive_tree_traversal(self, js_support, temp_project):
        """Test complex recursive tree traversal with multiple recursive calls."""
        code = """\
function traversePreOrder(node, visit) {
    if (!node) return;
    visit(node.value);
    traversePreOrder(node.left, visit);
    traversePreOrder(node.right, visit);
}

function traverseInOrder(node, visit) {
    if (!node) return;
    traverseInOrder(node.left, visit);
    visit(node.value);
    traverseInOrder(node.right, visit);
}

function traversePostOrder(node, visit) {
    if (!node) return;
    traversePostOrder(node.left, visit);
    traversePostOrder(node.right, visit);
    visit(node.value);
}

function collectAllValues(root) {
    const values = { pre: [], in: [], post: [] };

    traversePreOrder(root, v => values.pre.push(v));
    traverseInOrder(root, v => values.in.push(v));
    traversePostOrder(root, v => values.post.push(v));

    return values;
}
"""
        file_path = temp_project / "tree.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        collect_func = next(f for f in functions if f.function_name == "collectAllValues")

        context = js_support.extract_code_context(collect_func, temp_project, temp_project)

        expected_target_code = """\
function collectAllValues(root) {
    const values = { pre: [], in: [], post: [] };

    traversePreOrder(root, v => values.pre.push(v));
    traverseInOrder(root, v => values.in.push(v));
    traversePostOrder(root, v => values.post.push(v));

    return values;
}
"""
        assert context.target_code == expected_target_code

        # All traversal functions should be helpers
        helper_names = sorted([h.name for h in context.helper_functions])
        assert helper_names == ["traverseInOrder", "traversePostOrder", "traversePreOrder"]


class TestAsyncPatternsAndPromises:
    """Tests for async/await and Promise patterns."""

    def test_async_function_chain(self, js_support, temp_project):
        """Test async function that calls other async functions."""
        code = """\
async function fetchUserById(id) {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
        throw new Error(`User ${id} not found`);
    }
    return response.json();
}

async function fetchUserPosts(userId) {
    const response = await fetch(`/api/users/${userId}/posts`);
    return response.json();
}

async function fetchUserComments(userId) {
    const response = await fetch(`/api/users/${userId}/comments`);
    return response.json();
}

async function fetchUserProfile(userId) {
    const user = await fetchUserById(userId);
    const [posts, comments] = await Promise.all([
        fetchUserPosts(userId),
        fetchUserComments(userId)
    ]);

    return {
        ...user,
        posts,
        comments,
        totalActivity: posts.length + comments.length
    };
}
"""
        file_path = temp_project / "api.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        profile_func = next(f for f in functions if f.function_name == "fetchUserProfile")

        context = js_support.extract_code_context(profile_func, temp_project, temp_project)

        expected_target_code = """\
async function fetchUserProfile(userId) {
    const user = await fetchUserById(userId);
    const [posts, comments] = await Promise.all([
        fetchUserPosts(userId),
        fetchUserComments(userId)
    ]);

    return {
        ...user,
        posts,
        comments,
        totalActivity: posts.length + comments.length
    };
}
"""
        assert context.target_code == expected_target_code

        # All three async helpers should be found
        helper_names = sorted([h.name for h in context.helper_functions])
        assert helper_names == ["fetchUserById", "fetchUserComments", "fetchUserPosts"]


class TestExtractionReplacementRoundTrip:
    """Tests for full workflow of extracting and replacing code."""

    def test_extract_and_replace_class_method(self, js_support, temp_project):
        """Test extracting code context and then replacing the method."""
        original_source = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }

    decrement() {
        this.count--;
        return this.count;
    }
}

module.exports = { Counter };
"""
        file_path = temp_project / "counter.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        increment_func = next(fn for fn in functions if fn.function_name == "increment")

        # Step 1: Extract code context
        context = js_support.extract_code_context(increment_func, temp_project, temp_project)

        expected_extraction = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }
}
"""
        assert context.target_code == expected_extraction

        # Step 2: Simulate AI returning optimized code
        optimized_code_from_ai = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        return ++this.count;
    }
}
"""

        # Step 3: Replace in original
        result = js_support.replace_function(original_source, increment_func, optimized_code_from_ai)

        expected_result = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        return ++this.count;
    }

    decrement() {
        this.count--;
        return this.count;
    }
}

module.exports = { Counter };
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_function_with_complex_destructuring(self, js_support, temp_project):
        """Test function with complex nested destructuring parameters."""
        code = """\
function processApiResponse({
    data: { users = [], meta: { total, page } = {} } = {},
    status,
    headers: { 'content-type': contentType } = {}
}) {
    return {
        users,
        pagination: { total, page },
        status,
        contentType
    };
}
"""
        file_path = temp_project / "api.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function processApiResponse({
    data: { users = [], meta: { total, page } = {} } = {},
    status,
    headers: { 'content-type': contentType } = {}
}) {
    return {
        users,
        pagination: { total, page },
        status,
        contentType
    };
}
"""
        assert context.target_code == expected_target_code
        assert js_support.validate_syntax(context.target_code) is True

    def test_generator_function(self, js_support, temp_project):
        """Test generator function extraction."""
        code = """\
function* range(start, end, step = 1) {
    for (let i = start; i < end; i += step) {
        yield i;
    }
}

function* fibonacci(limit) {
    let [a, b] = [0, 1];
    while (a < limit) {
        yield a;
        [a, b] = [b, a + b];
    }
}
"""
        file_path = temp_project / "generators.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        range_func = next(f for f in functions if f.function_name == "range")

        context = js_support.extract_code_context(range_func, temp_project, temp_project)

        expected_target_code = """\
function* range(start, end, step = 1) {
    for (let i = start; i < end; i += step) {
        yield i;
    }
}
"""
        assert context.target_code == expected_target_code
        assert context.helper_functions == []

    def test_function_with_computed_property_names(self, js_support, temp_project):
        """Test function returning object with computed property names."""
        code = """\
const FIELD_KEYS = {
    NAME: 'user_name',
    EMAIL: 'user_email',
    AGE: 'user_age'
};

function createUserObject(name, email, age) {
    return {
        [FIELD_KEYS.NAME]: name,
        [FIELD_KEYS.EMAIL]: email,
        [FIELD_KEYS.AGE]: age
    };
}
"""
        file_path = temp_project / "user.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        context = js_support.extract_code_context(func, temp_project, temp_project)

        expected_target_code = """\
function createUserObject(name, email, age) {
    return {
        [FIELD_KEYS.NAME]: name,
        [FIELD_KEYS.EMAIL]: email,
        [FIELD_KEYS.AGE]: age
    };
}
"""
        assert context.target_code == expected_target_code

        # Exact match for read_only_context
        expected_read_only = """\
const FIELD_KEYS = {
    NAME: 'user_name',
    EMAIL: 'user_email',
    AGE: 'user_age'
};"""
        assert context.read_only_context == expected_read_only

    def test_with_tricky_helpers(self, ts_support, temp_project):
        """Test function returning object with computed property names."""
        code = """import { WebClient, ChatPostMessageArguments } from "@slack/web-api"

// Dependencies interface for easier testing
export interface SendSlackMessageDependencies {
  WebClient: typeof WebClient
  getSlackToken: () => string | undefined
  getSlackChannelId: () => string | undefined
  console: typeof console
}

// Default dependencies
let dependencies: SendSlackMessageDependencies = {
  WebClient,
  getSlackToken: () => process.env.SLACK_TOKEN,
  getSlackChannelId: () => process.env.SLACK_CHANNEL_ID,
  console,
}

// For testing - allow dependency injection
export function setSendSlackMessageDependencies(deps: Partial<SendSlackMessageDependencies>) {
  dependencies = { ...dependencies, ...deps }
}

export function resetSendSlackMessageDependencies() {
  dependencies = {
    WebClient,
    getSlackToken: () => process.env.SLACK_TOKEN,
    getSlackChannelId: () => process.env.SLACK_CHANNEL_ID,
    console,
  }
}

// Initialize web client
let web: WebClient | null = null

export function initializeWebClient() {
  const SLACK_TOKEN = dependencies.getSlackToken()
  const SLACK_CHANNEL_ID = dependencies.getSlackChannelId()

  if (!SLACK_TOKEN) {
    throw new Error("Missing SLACK_TOKEN")
  }

  if (!SLACK_CHANNEL_ID) {
    throw new Error("Missing SLACK_CHANNEL_ID")
  }

  if (!web) {
    web = new dependencies.WebClient(SLACK_TOKEN, {})
  }

  return web
}

// For testing - allow resetting the web client
export function resetWebClient() {
  web = null
}

/**
 * Send a message to Slack
 *
 * @param {string|object} message - Text message or Block Kit message object
 * @param {string|null} channel - Channel ID, defaults to SLACK_CHANNEL_ID
 * @param {boolean} returnData - Whether to return the full Slack API response
 * @returns {Promise<boolean|object>} - True or API response
 */
export const sendSlackMessage = async (
  message: any,
  channel: string | null = null,
  returnData: boolean = false,
): Promise<boolean | object> => {
  return new Promise(async (resolve, reject) => {
    try {
      const webClient = initializeWebClient()
      const SLACK_CHANNEL_ID = dependencies.getSlackChannelId()
      const channelId = channel || SLACK_CHANNEL_ID

      // Configure the message payload depending on the input type
      let payload: ChatPostMessageArguments

      if (typeof message === "string") {
        payload = {
          channel: channelId,
          text: message,
        }
      } else if (message && typeof message === "object") {
        if (message.blocks) {
          payload = {
            channel: channelId,
            text: message.text || "Notification from CodeFlash",
            blocks: message.blocks,
          }
        } else {
          dependencies.console.warn("Object passed to sendSlackMessage without blocks property")
          payload = {
            channel: channelId,
            text: JSON.stringify(message),
          }
        }
      } else {
        dependencies.console.error("Invalid message type", typeof message)
        payload = {
          channel: channelId,
          text: "Invalid message",
        }
      }

      // console.log("Sending payload to Slack:", JSON.stringify(payload, null, 2));

      const resp = await webClient.chat.postMessage(payload)
      return resolve(returnData ? resp : true)
    } catch (error) {
      dependencies.console.error("Error sending Slack message:", error)
      return resolve(returnData ? { error } : true)
    }
  })
}
"""
        file_path = temp_project / "slack_util.ts"
        file_path.write_text(code, encoding="utf-8")
        target_func = "sendSlackMessage"

        functions = ts_support.discover_functions(file_path)
        func_info = next(f for f in functions if f.function_name == target_func)
        fto = FunctionToOptimize(
            function_name=target_func,
            file_path=file_path,
            parents=func_info.parents,
            starting_line=func_info.start_line,
            ending_line=func_info.end_line,
            starting_col=func_info.start_col,
            ending_col=func_info.end_col,
            is_async=func_info.is_async,
            language="typescript",
        )

        ctx = get_code_optimization_context_for_language(
            fto, temp_project
        )

        # The read_writable_code should contain the target function AND helper functions
        expected_read_writable = """```typescript:slack_util.ts
import { WebClient, ChatPostMessageArguments } from "@slack/web-api"

export const sendSlackMessage = async (
  message: any,
  channel: string | null = null,
  returnData: boolean = false,
): Promise<boolean | object> => {
  return new Promise(async (resolve, reject) => {
    try {
      const webClient = initializeWebClient()
      const SLACK_CHANNEL_ID = dependencies.getSlackChannelId()
      const channelId = channel || SLACK_CHANNEL_ID

      // Configure the message payload depending on the input type
      let payload: ChatPostMessageArguments

      if (typeof message === "string") {
        payload = {
          channel: channelId,
          text: message,
        }
      } else if (message && typeof message === "object") {
        if (message.blocks) {
          payload = {
            channel: channelId,
            text: message.text || "Notification from CodeFlash",
            blocks: message.blocks,
          }
        } else {
          dependencies.console.warn("Object passed to sendSlackMessage without blocks property")
          payload = {
            channel: channelId,
            text: JSON.stringify(message),
          }
        }
      } else {
        dependencies.console.error("Invalid message type", typeof message)
        payload = {
          channel: channelId,
          text: "Invalid message",
        }
      }

      // console.log("Sending payload to Slack:", JSON.stringify(payload, null, 2));

      const resp = await webClient.chat.postMessage(payload)
      return resolve(returnData ? resp : true)
    } catch (error) {
      dependencies.console.error("Error sending Slack message:", error)
      return resolve(returnData ? { error } : true)
    }
  })
}


export function initializeWebClient() {
  const SLACK_TOKEN = dependencies.getSlackToken()
  const SLACK_CHANNEL_ID = dependencies.getSlackChannelId()

  if (!SLACK_TOKEN) {
    throw new Error("Missing SLACK_TOKEN")
  }

  if (!SLACK_CHANNEL_ID) {
    throw new Error("Missing SLACK_CHANNEL_ID")
  }

  if (!web) {
    web = new dependencies.WebClient(SLACK_TOKEN, {})
  }

  return web
}
```"""

        # The read_only_context should contain global variables (dependencies object, web client)
        # but NOT have invalid floating object properties
        expected_read_only = """let dependencies: SendSlackMessageDependencies = {
  WebClient,
  getSlackToken: () => process.env.SLACK_TOKEN,
  getSlackChannelId: () => process.env.SLACK_CHANNEL_ID,
  console,
}
let web: WebClient | null = null"""

        assert ctx.read_writable_code.markdown == expected_read_writable
        assert ctx.read_only_context_code == expected_read_only



class TestContextProperties:
    """Tests for CodeContext object properties."""

    def test_javascript_context_has_correct_language(self, js_support, temp_project):
        """Test that JavaScript context has correct language property."""
        code = """\
function test() {
    return 1;
}
"""
        file_path = temp_project / "test.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        context = js_support.extract_code_context(functions[0], temp_project, temp_project)

        assert context.language == Language.JAVASCRIPT
        assert context.target_file == file_path
        assert context.helper_functions == []
        assert context.read_only_context == ""
        assert isinstance(context.imports, list)

    def test_typescript_context_has_javascript_language(self, ts_support, temp_project):
        """Test that TypeScript context uses JavaScript language enum."""
        code = """\
function test(): number {
    return 1;
}
"""
        file_path = temp_project / "test.ts"
        file_path.write_text(code, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        context = ts_support.extract_code_context(functions[0], temp_project, temp_project)

        # TypeScript uses JavaScript language enum
        assert context.language == Language.JAVASCRIPT
        assert context.target_file == file_path


class TestContextValidation:
    """Tests to verify extracted context produces valid syntax."""

    def test_all_class_methods_produce_valid_syntax(self, js_support, temp_project):
        """Test that all extracted class methods are syntactically valid JavaScript."""
        code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return Number((a + b).toFixed(this.precision));
    }

    subtract(a, b) {
        return Number((a - b).toFixed(this.precision));
    }

    multiply(a, b) {
        return Number((a * b).toFixed(this.precision));
    }

    divide(a, b) {
        if (b === 0) {
            throw new Error('Division by zero');
        }
        return Number((a / b).toFixed(this.precision));
    }
}
"""
        file_path = temp_project / "calculator.js"
        file_path.write_text(code, encoding="utf-8")

        functions = js_support.discover_functions(file_path)

        for func in functions:
            if func.function_name != "constructor":
                context = js_support.extract_code_context(func, temp_project, temp_project)
                is_valid = js_support.validate_syntax(context.target_code)
                assert is_valid is True, f"Invalid syntax for {func.name}:\n{context.target_code}"
