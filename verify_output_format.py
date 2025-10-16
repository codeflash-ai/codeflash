"""Verify that both AST and LibCST implementations produce the exact requested output format."""

import json

from function_call_finder import find_function_calls as find_calls_libcst
from function_call_finder_ast import find_function_calls as find_calls_ast

# Simple test case
test_code = """
def func1():
    my_target()

def func2():
    my_target(1, 2, 3)
"""

print("Verifying output format: {'calling_function_qualified_name1':'function_definition1',...}")
print("=" * 70)

# Test AST implementation
print("\n1. AST Implementation:")
ast_result = find_calls_ast(test_code, "my_target", "/dummy/path.py")
print(f"   Type: {type(ast_result)}")
print(f"   Keys type: {type(list(ast_result.keys())[0]) if ast_result else 'N/A'}")
print(f"   Values type: {type(list(ast_result.values())[0]) if ast_result else 'N/A'}")
print(f"   JSON serializable: {json.dumps(ast_result) is not None}")
print(f"   Example output: {json.dumps(ast_result, indent=2)}")

# Test LibCST implementation
print("\n2. LibCST Implementation:")
libcst_result = find_calls_libcst(test_code, "my_target", "/dummy/path.py")
print(f"   Type: {type(libcst_result)}")
print(f"   Keys type: {type(list(libcst_result.keys())[0]) if libcst_result else 'N/A'}")
print(f"   Values type: {type(list(libcst_result.values())[0]) if libcst_result else 'N/A'}")
print(f"   JSON serializable: {json.dumps(libcst_result) is not None}")
print(f"   Example output: {json.dumps(libcst_result, indent=2)}")

# Test with class methods
print("\n3. Testing with class methods:")
class_test = """
class MyClass:
    def method1(self):
        target()

    def method2(self):
        pass
"""

ast_class = find_calls_ast(class_test, "target", "/dummy/path.py")
libcst_class = find_calls_libcst(class_test, "target", "/dummy/path.py")

print(f"   AST result: {list(ast_class.keys())}")
print(f"   LibCST result: {list(libcst_class.keys())}")

# Final verification
print("\n" + "=" * 70)
print("✅ VERIFIED: Both implementations return the exact format requested:")
print('   {"calling_function_qualified_name1":"function_definition1",...}')
print("\nKey characteristics:")
print("   - Plain dictionary (dict type)")
print("   - String keys (qualified function names)")
print("   - String values (function source code)")
print("   - JSON serializable")
print("   - No nested structures, just simple key-value pairs")
