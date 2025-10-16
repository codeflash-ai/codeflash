#!/usr/bin/env python3
"""Script to find all references to the sorter function from code_to_optimize/bubble_sort.py
using Jedi's static analysis capabilities.
"""

from pathlib import Path

import jedi


def find_function_references(file_path, line, column, project_root):
    """Find all references to a function using Jedi.

    Args:
        file_path: Path to the file containing the function
        line: Line number where the function is defined (1-indexed)
        column: Column number where the function name starts (0-indexed)
        project_root: Root directory of the project to search

    """
    # Read the source code
    with open(file_path) as f:
        source = f.read()

    # Create a Jedi Script object with project configuration
    project = jedi.Project(path=project_root)
    script = jedi.Script(source, path=file_path, project=project)

    # Get the function definition at the specified position
    definitions = script.goto(line, column, follow_imports=True)

    if not definitions:
        print(f"No definition found at {file_path}:{line}:{column}")
        return []

    # Get the first definition (should be the function itself)
    definition = definitions[0]
    print(f"Found definition: {definition.name} at {definition.module_path}:{definition.line}")
    print(f"Type: {definition.type}")
    print("-" * 80)

    # Use search_all to find all references to this function
    # We'll search for references by name throughout the project
    references = []
    try:
        # Use usages() method to get all references
        references = script.get_references(line, column, scope="project", include_builtins=False)
    except AttributeError:
        # Alternative approach using search
        print("Using alternative search method...")
        references = script.get_references(line, column, include_builtins=False)

    return references


def main():
    # Project root directory
    project_root = Path("/Users/aseemsaxena/Downloads/codeflash_dev/codeflash")

    # Target file and function location
    target_file = project_root / "code_to_optimize" / "bubble_sort.py"

    # The sorter function starts at line 1, column 4 (0-indexed)
    # "def sorter(arr):" - the function name 'sorter' starts at column 4
    line = 1  # Line number (1-indexed)
    column = 4  # Column number (0-indexed) - position of 's' in 'sorter'

    print(f"Searching for references to 'sorter' function in {target_file}")
    print(f"Position: Line {line}, Column {column}")
    print("=" * 80)

    # Find references
    references = find_function_references(target_file, line, column, project_root)

    if references:
        print(f"\nFound {len(references)} reference(s) to 'sorter' function:")
        print("=" * 80)

        # Group references by file
        refs_by_file = {}
        for ref in references:
            file_path = ref.module_path
            if file_path not in refs_by_file:
                refs_by_file[file_path] = []
            refs_by_file[file_path].append(ref)

        # Display references organized by file
        for file_path, file_refs in sorted(refs_by_file.items()):
            print(f"\n📁 {file_path}")
            for ref in sorted(file_refs, key=lambda r: (r.line, r.column)):
                # Get the line content for context
                try:
                    with open(file_path) as f:
                        lines = f.readlines()
                        if ref.line <= len(lines):
                            line_content = lines[ref.line - 1].strip()
                            print(f"  Line {ref.line}, Col {ref.column}: {line_content}")
                        else:
                            print(f"  Line {ref.line}, Col {ref.column}")
                except Exception as e:
                    print(f"  Line {ref.line}, Col {ref.column} (couldn't read line: {e})")
    else:
        print("\nNo references found to the 'sorter' function.")

    print("\n" + "=" * 80)
    print("Search complete!")


if __name__ == "__main__":
    main()
