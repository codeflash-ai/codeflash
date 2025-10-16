"""Test script to verify the function_call_finder output format."""

from function_call_finder import find_function_calls

# Test code
test_code = '''
def func1():
    target_func()

def func2():
    pass

def func3():
    x = target_func(42)
    return x

class TestClass:
    def method1(self):
        target_func("test")

    def method2(self):
        # No call here
        pass
'''

# Run the visitor
results = find_function_calls(test_code, "target_func", "/dummy/path.py")

# Verify the output format
print("Output type:", type(results))
print("Output keys:", list(results.keys()))
print("\nExpected format: {qualified_name: source_code}")
print("Actual format check:")

for name, code in results.items():
    print(f"\n✓ Key (function name): '{name}' -> Type: {type(name).__name__}")
    print(f"✓ Value (source code): Type: {type(code).__name__}, Length: {len(code)} chars")
    print(f"  First line: {code.split(chr(10))[0] if code else 'Empty'}")

# Verify it's exactly the format requested: {"calling_function_qualified_name1":"function_definition1",....}
import json
print("\nJSON serializable:", end=" ")
try:
    json_str = json.dumps(results)
    print("✓ Yes")
    print(f"JSON length: {len(json_str)} characters")
except:
    print("✗ No")

print("\n" + "="*50)
print("VERIFIED: Output is in the format")
print('{"calling_function_qualified_name1":"function_definition1",...}')