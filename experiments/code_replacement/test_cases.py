"""
Test cases for evaluating JavaScript/TypeScript code replacement strategies.

Each test case includes:
- original_source: The original JS/TS code
- function_name: Name of the function to replace
- start_line, end_line: Line numbers of the function (1-indexed)
- new_function: The replacement function code
- expected_result: What the output should look like
- description: What edge case this tests
"""

from dataclasses import dataclass


@dataclass
class ReplacementTestCase:
    name: str
    description: str
    original_source: str
    function_name: str
    start_line: int
    end_line: int
    new_function: str
    expected_result: str


# Test cases covering various JavaScript/TypeScript patterns
TEST_CASES = [
    # ===========================================
    # BASIC CASES
    # ===========================================
    ReplacementTestCase(
        name="simple_function",
        description="Basic named function declaration",
        original_source='''function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
''',
        function_name="add",
        start_line=1,
        end_line=3,
        new_function='''function add(a, b) {
    // Optimized version
    return a + b | 0;
}''',
        expected_result='''function add(a, b) {
    // Optimized version
    return a + b | 0;
}

function multiply(a, b) {
    return a * b;
}
'''
    ),

    ReplacementTestCase(
        name="arrow_function_const",
        description="Arrow function assigned to const",
        original_source='''const square = (x) => {
    return x * x;
};

const cube = (x) => x * x * x;
''',
        function_name="square",
        start_line=1,
        end_line=3,
        new_function='''const square = (x) => {
    return x ** 2;
};''',
        expected_result='''const square = (x) => {
    return x ** 2;
};

const cube = (x) => x * x * x;
'''
    ),

    ReplacementTestCase(
        name="arrow_function_oneliner",
        description="Single-line arrow function",
        original_source='''const double = x => x * 2;
const triple = x => x * 3;
''',
        function_name="double",
        start_line=1,
        end_line=1,
        new_function='''const double = x => x << 1;''',
        expected_result='''const double = x => x << 1;
const triple = x => x * 3;
'''
    ),

    # ===========================================
    # CLASS METHODS
    # ===========================================
    ReplacementTestCase(
        name="class_method",
        description="Method inside a class",
        original_source='''class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(n) {
        return this.value + n;
    }

    multiply(n) {
        return this.value * n;
    }
}
''',
        function_name="add",
        start_line=6,
        end_line=8,
        new_function='''    add(n) {
        // Optimized addition
        return (this.value + n) | 0;
    }''',
        expected_result='''class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(n) {
        // Optimized addition
        return (this.value + n) | 0;
    }

    multiply(n) {
        return this.value * n;
    }
}
'''
    ),

    ReplacementTestCase(
        name="static_method",
        description="Static method in class",
        original_source='''class MathUtils {
    static fibonacci(n) {
        if (n <= 1) return n;
        return MathUtils.fibonacci(n - 1) + MathUtils.fibonacci(n - 2);
    }

    static factorial(n) {
        if (n <= 1) return 1;
        return n * MathUtils.factorial(n - 1);
    }
}
''',
        function_name="fibonacci",
        start_line=2,
        end_line=5,
        new_function='''    static fibonacci(n) {
        // Memoized version
        const memo = [0, 1];
        for (let i = 2; i <= n; i++) {
            memo[i] = memo[i-1] + memo[i-2];
        }
        return memo[n];
    }''',
        expected_result='''class MathUtils {
    static fibonacci(n) {
        // Memoized version
        const memo = [0, 1];
        for (let i = 2; i <= n; i++) {
            memo[i] = memo[i-1] + memo[i-2];
        }
        return memo[n];
    }

    static factorial(n) {
        if (n <= 1) return 1;
        return n * MathUtils.factorial(n - 1);
    }
}
'''
    ),

    # ===========================================
    # ASYNC FUNCTIONS
    # ===========================================
    ReplacementTestCase(
        name="async_function",
        description="Async function declaration",
        original_source='''async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}

async function postData(url, data) {
    const response = await fetch(url, { method: 'POST', body: JSON.stringify(data) });
    return response.json();
}
''',
        function_name="fetchData",
        start_line=1,
        end_line=4,
        new_function='''async function fetchData(url) {
    // With caching
    const cached = cache.get(url);
    if (cached) return cached;
    const response = await fetch(url);
    const data = await response.json();
    cache.set(url, data);
    return data;
}''',
        expected_result='''async function fetchData(url) {
    // With caching
    const cached = cache.get(url);
    if (cached) return cached;
    const response = await fetch(url);
    const data = await response.json();
    cache.set(url, data);
    return data;
}

async function postData(url, data) {
    const response = await fetch(url, { method: 'POST', body: JSON.stringify(data) });
    return response.json();
}
'''
    ),

    # ===========================================
    # EDGE CASES: COMMENTS & WHITESPACE
    # ===========================================
    ReplacementTestCase(
        name="function_with_jsdoc",
        description="Function with JSDoc comment above it",
        original_source='''/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function sum(a, b) {
    return a + b;
}

function diff(a, b) {
    return a - b;
}
''',
        function_name="sum",
        start_line=7,  # Function starts after JSDoc
        end_line=9,
        new_function='''function sum(a, b) {
    return (a + b) | 0;
}''',
        expected_result='''/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function sum(a, b) {
    return (a + b) | 0;
}

function diff(a, b) {
    return a - b;
}
'''
    ),

    ReplacementTestCase(
        name="inline_comments",
        description="Function with inline comments",
        original_source='''function process(data) {
    // Validate input
    if (!data) return null;

    // Transform data
    const result = data.map(x => x * 2); // double each value

    return result;
}
''',
        function_name="process",
        start_line=1,
        end_line=9,
        new_function='''function process(data) {
    if (!data) return null;
    return data.map(x => x << 1);
}''',
        expected_result='''function process(data) {
    if (!data) return null;
    return data.map(x => x << 1);
}
'''
    ),

    # ===========================================
    # NESTED FUNCTIONS
    # ===========================================
    ReplacementTestCase(
        name="function_with_nested",
        description="Function containing nested functions",
        original_source='''function outer(x) {
    function inner(y) {
        return y * 2;
    }
    return inner(x) + 1;
}

function other() {
    return 42;
}
''',
        function_name="outer",
        start_line=1,
        end_line=6,
        new_function='''function outer(x) {
    const inner = y => y << 1;
    return inner(x) + 1;
}''',
        expected_result='''function outer(x) {
    const inner = y => y << 1;
    return inner(x) + 1;
}

function other() {
    return 42;
}
'''
    ),

    # ===========================================
    # TYPESCRIPT SPECIFIC
    # ===========================================
    ReplacementTestCase(
        name="typescript_typed_function",
        description="TypeScript function with type annotations",
        original_source='''function greet(name: string): string {
    return `Hello, ${name}!`;
}

function farewell(name: string): string {
    return `Goodbye, ${name}!`;
}
''',
        function_name="greet",
        start_line=1,
        end_line=3,
        new_function='''function greet(name: string): string {
    return 'Hello, ' + name + '!';
}''',
        expected_result='''function greet(name: string): string {
    return 'Hello, ' + name + '!';
}

function farewell(name: string): string {
    return `Goodbye, ${name}!`;
}
'''
    ),

    ReplacementTestCase(
        name="typescript_generic",
        description="TypeScript generic function",
        original_source='''function identity<T>(arg: T): T {
    return arg;
}

function first<T>(arr: T[]): T | undefined {
    return arr[0];
}
''',
        function_name="identity",
        start_line=1,
        end_line=3,
        new_function='''function identity<T>(arg: T): T {
    // Direct return
    return arg;
}''',
        expected_result='''function identity<T>(arg: T): T {
    // Direct return
    return arg;
}

function first<T>(arr: T[]): T | undefined {
    return arr[0];
}
'''
    ),

    ReplacementTestCase(
        name="typescript_interface_method",
        description="TypeScript class implementing interface",
        original_source='''interface Processor {
    process(data: number[]): number[];
}

class ArrayProcessor implements Processor {
    process(data: number[]): number[] {
        return data.map(x => x * 2);
    }

    transform(data: number[]): number[] {
        return data.filter(x => x > 0);
    }
}
''',
        function_name="process",
        start_line=6,
        end_line=8,
        new_function='''    process(data: number[]): number[] {
        const result = new Array(data.length);
        for (let i = 0; i < data.length; i++) {
            result[i] = data[i] << 1;
        }
        return result;
    }''',
        expected_result='''interface Processor {
    process(data: number[]): number[];
}

class ArrayProcessor implements Processor {
    process(data: number[]): number[] {
        const result = new Array(data.length);
        for (let i = 0; i < data.length; i++) {
            result[i] = data[i] << 1;
        }
        return result;
    }

    transform(data: number[]): number[] {
        return data.filter(x => x > 0);
    }
}
'''
    ),

    # ===========================================
    # EXPORT PATTERNS
    # ===========================================
    ReplacementTestCase(
        name="exported_function",
        description="Exported function declaration",
        original_source='''export function calculate(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}
''',
        function_name="calculate",
        start_line=1,
        end_line=3,
        new_function='''export function calculate(a, b) {
    return (a + b) | 0;
}''',
        expected_result='''export function calculate(a, b) {
    return (a + b) | 0;
}

export function subtract(a, b) {
    return a - b;
}
'''
    ),

    ReplacementTestCase(
        name="default_export",
        description="Default exported function",
        original_source='''export default function main(args) {
    return args.reduce((a, b) => a + b, 0);
}

function helper(x) {
    return x * 2;
}
''',
        function_name="main",
        start_line=1,
        end_line=3,
        new_function='''export default function main(args) {
    let sum = 0;
    for (const arg of args) sum += arg;
    return sum;
}''',
        expected_result='''export default function main(args) {
    let sum = 0;
    for (const arg of args) sum += arg;
    return sum;
}

function helper(x) {
    return x * 2;
}
'''
    ),

    # ===========================================
    # DECORATORS (TypeScript/Experimental JS)
    # ===========================================
    ReplacementTestCase(
        name="decorated_method",
        description="Method with decorators",
        original_source='''class Service {
    @log
    @memoize
    compute(x: number): number {
        return x * x;
    }

    other(): void {
        console.log('other');
    }
}
''',
        function_name="compute",
        start_line=4,  # Method starts after decorators
        end_line=6,
        new_function='''    compute(x: number): number {
        return x ** 2;
    }''',
        expected_result='''class Service {
    @log
    @memoize
    compute(x: number): number {
        return x ** 2;
    }

    other(): void {
        console.log('other');
    }
}
'''
    ),

    # ===========================================
    # FIRST/LAST FUNCTION EDGE CASES
    # ===========================================
    ReplacementTestCase(
        name="first_function_in_file",
        description="Replacing the very first function in file",
        original_source='''function first() {
    return 1;
}

function second() {
    return 2;
}
''',
        function_name="first",
        start_line=1,
        end_line=3,
        new_function='''function first() {
    return 1 | 0;
}''',
        expected_result='''function first() {
    return 1 | 0;
}

function second() {
    return 2;
}
'''
    ),

    ReplacementTestCase(
        name="last_function_in_file",
        description="Replacing the last function in file",
        original_source='''function first() {
    return 1;
}

function last() {
    return 999;
}
''',
        function_name="last",
        start_line=5,
        end_line=7,
        new_function='''function last() {
    return 1000;
}''',
        expected_result='''function first() {
    return 1;
}

function last() {
    return 1000;
}
'''
    ),

    ReplacementTestCase(
        name="only_function_in_file",
        description="Replacing the only function in file",
        original_source='''function only() {
    return 42;
}
''',
        function_name="only",
        start_line=1,
        end_line=3,
        new_function='''function only() {
    return 42 | 0;
}''',
        expected_result='''function only() {
    return 42 | 0;
}
'''
    ),

    # ===========================================
    # INDENTATION PRESERVATION
    # ===========================================
    ReplacementTestCase(
        name="deeply_nested_method",
        description="Method with deep indentation",
        original_source='''const module = {
    submodule: {
        handler: {
            process(data) {
                return data.map(x => x * 2);
            }
        }
    }
};
''',
        function_name="process",
        start_line=4,
        end_line=6,
        new_function='''            process(data) {
                return data.map(x => x << 1);
            }''',
        expected_result='''const module = {
    submodule: {
        handler: {
            process(data) {
                return data.map(x => x << 1);
            }
        }
    }
};
'''
    ),
]


def get_test_cases():
    """Return all test cases."""
    return TEST_CASES


def get_test_case_by_name(name: str) -> ReplacementTestCase | None:
    """Get a specific test case by name."""
    for tc in TEST_CASES:
        if tc.name == name:
            return tc
    return None
