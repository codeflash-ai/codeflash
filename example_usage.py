#!/usr/bin/env python3
"""Example usage of the function reference finder."""

from find_function_references import find_function_references
from find_function_references_detailed import find_function_references_detailed


def example_basic_usage():
    """Example of basic usage - just get list of files."""
    # Example: Find references to a function
    filepath = "path/to/your/file.py"
    function_name = "your_function_name"

    # Find references (will auto-detect project root)
    reference_files = find_function_references(filepath, function_name)

    print(f"Files containing references to {function_name}:")
    for file in reference_files:
        print(f"  - {file}")

    return reference_files


def example_detailed_usage():
    """Example of detailed usage - get line numbers and context."""
    # Example: Find references with detailed information
    filepath = "path/to/your/file.py"
    function_name = "your_function_name"
    project_root = "/path/to/project/root"  # Optional

    # Find detailed references
    references = find_function_references_detailed(filepath, function_name, project_root)

    print(f"\nDetailed references to {function_name}:")
    for file, refs in references.items():
        print(f"\nFile: {file}")
        for ref in refs:
            print(f"  Line {ref['line']}: {ref['context']}")

    return references


def example_programmatic_usage():
    """Example of using in your own Python code."""
    import jedi

    # Direct Jedi usage for more control
    source_code = """
def my_function(x, y):
    return x + y

result = my_function(1, 2)
"""

    # Create a script object
    script = jedi.Script(code=source_code)

    # Find the function definition (line 2, column 4 for 'my_function')
    references = script.get_references(line=2, column=4)

    print("Direct Jedi references:")
    for ref in references:
        print(f"  Line {ref.line}, Column {ref.column}: {ref.description}")

    # You can also search for names
    names = script.get_names()
    for name in names:
        if name.type == "function":
            print(f"Found function: {name.name} at line {name.line}")


def example_find_all_functions_and_their_references():
    """Example of finding all functions in a file and their references."""
    import os

    import jedi

    def find_all_functions_with_references(filepath: str):
        """Find all functions in a file and their references."""
        with open(filepath) as f:
            source_code = f.read()

        script = jedi.Script(code=source_code, path=filepath)

        # Get all defined names
        names = script.get_names()

        functions_and_refs = {}

        for name in names:
            if name.type == "function":
                # Get references for this function
                refs = script.get_references(line=name.line, column=name.column)

                ref_locations = []
                for ref in refs:
                    if ref.module_path and str(ref.module_path) != filepath:
                        ref_locations.append({"file": str(ref.module_path), "line": ref.line, "column": ref.column})

                functions_and_refs[name.name] = {"definition_line": name.line, "references": ref_locations}

        return functions_and_refs

    # Example usage
    filepath = "your_module.py"
    if os.path.exists(filepath):
        all_refs = find_all_functions_with_references(filepath)

        for func_name, info in all_refs.items():
            print(f"\nFunction: {func_name} (defined at line {info['definition_line']})")
            if info["references"]:
                print("  Referenced in:")
                for ref in info["references"]:
                    print(f"    - {ref['file']}:{ref['line']}")
            else:
                print("  No external references found")


if __name__ == "__main__":
    print("=" * 60)
    print("Function Reference Finder - Usage Examples")
    print("=" * 60)

    print("\nNote: Update the file paths and function names in the examples")
    print("before running them with your actual code.\n")

    # Uncomment to run examples:
    # example_basic_usage()
    # example_detailed_usage()
    # example_programmatic_usage()
    # example_find_all_functions_and_their_references()
