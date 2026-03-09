from codeflash.languages.javascript.normalizer import normalize_js_code
from codeflash.languages.python.normalizer import normalize_python_code as normalize_code


def test_deduplicate1():
    # Example usage and tests
    # Example 1: Same logic, different variable names (should NOT match due to different function/param names)
    code1 = """
def compute_sum(numbers):
    '''Calculate sum of numbers'''
    total = 0
    for num in numbers:
        total += num
    return total
"""

    code2 = """
def compute_sum(numbers):
    # This computes the sum
    result = 0
    for value in numbers:
        result += value
    return result
"""

    assert normalize_code(code1) == normalize_code(code2)
    assert normalize_code(code1) == normalize_code(code2)

    # Example 3: Same function and parameter names, different local variables (should match)
    code3 = """
def calculate_sum(numbers):
    accumulator = 0
    for item in numbers:
        accumulator += item
    return accumulator
"""

    code4 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

    assert normalize_code(code3) == normalize_code(code4)
    assert normalize_code(code3) == normalize_code(code4)

    # Example 4: Nested functions and classes (preserving names)
    code5 = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        def helper(item):
            temp = item * 2
            return temp
    
        results = []
        for element in self.data:
            results.append(helper(element))
        return results
"""

    code6 = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        def helper(item):
            x = item * 2
            return x
    
        output = []
        for thing in self.data:
            output.append(helper(thing))
        return output
"""

    assert normalize_code(code5) == normalize_code(code6)

    # Example 5: With imports and built-ins (these should be preserved)
    code7 = """
import math

def calculate_circle_area(radius):
    pi_value = math.pi
    area = pi_value * radius ** 2
    return area
"""

    code8 = """
import math

def calculate_circle_area(radius):
    constant = math.pi
    result = constant * radius ** 2
    return result
"""
    code85 = """
import math

def calculate_circle_area(radius):
    constant = math.pi
    result = constant *2 * radius ** 2
    return result
"""

    assert normalize_code(code7) == normalize_code(code8)
    assert normalize_code(code8) != normalize_code(code85)

    # Example 6: Exception handling
    code9 = """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError as e:
        error_msg = str(e)
        return None
"""

    code10 = """
def safe_divide(a, b):
    try:
        output = a / b
        return output
    except ZeroDivisionError as exc:
        message = str(exc)
        return None
"""
    assert normalize_code(code9) == normalize_code(code10)

    assert normalize_code(code9) != normalize_code(code8)


# === JavaScript deduplication tests ===


def test_js_deduplicate_same_logic_different_vars():
    code1 = """
function process(items) {
    const result = [];
    for (const item of items) {
        result.push(item * 2);
    }
    return result;
}
"""
    code2 = """
function process(items) {
    const output = [];
    for (const val of items) {
        output.push(val * 2);
    }
    return output;
}
"""
    assert normalize_js_code(code1) == normalize_js_code(code2)


def test_js_different_logic_not_deduplicated():
    code1 = """
function compute(x) {
    return x + 1;
}
"""
    code2 = """
function compute(x) {
    return x * 2;
}
"""
    assert normalize_js_code(code1) != normalize_js_code(code2)


def test_js_deduplicate_whitespace_and_comments():
    code1 = """
function add(a, b) {
    // fast path
    return a + b;
}
"""
    code2 = """
function add(a, b) {
    /* optimized */
    return a + b;
}
"""
    assert normalize_js_code(code1) == normalize_js_code(code2)


def test_ts_normalize():
    code1 = """
function greet(name: string): string {
    const msg = "hello " + name;
    return msg;
}
"""
    code2 = """
function greet(name: string): string {
    const result = "hello " + name;
    return result;
}
"""
    assert normalize_js_code(code1, typescript=True) == normalize_js_code(code2, typescript=True)
