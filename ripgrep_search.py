#!/usr/bin/env python3
"""
Script to find all occurrences of 'function_name' in the repository using ripgrep.
Returns a dictionary where keys are filepaths and values are lists of (line_no, content) tuples.
"""
import os
import subprocess
import json
from typing import Dict, List, Tuple
from pathlib import Path


def search_with_ripgrep(pattern: str, path: str = ".") -> Dict[str, List[Tuple[int, str]]]:
    """
    Use ripgrep to search for a pattern in the repository.

    Args:
        pattern: The pattern to search for
        path: The directory to search in (default: current directory)

    Returns:
        Dictionary with filepaths as keys and list of (line_no, content) tuples as values
    """
    # Run ripgrep with JSON output for easier parsing
    # -n: Show line numbers
    # --json: Output in JSON format
    # --no-heading: Don't group matches by file
    path = str(Path.cwd())
    cmd = ["rg", "-n", "--json", pattern, path, "-g", "!/Users/aseemsaxena/Downloads/codeflash_dev/codeflash/code_to_optimize/tests/**"]
    print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero return
        )

        if result.returncode not in [0, 1]:  # 0 = matches found, 1 = no matches
            print(f"Error running ripgrep: {result.stderr}")
            return {}

        # Parse the JSON output
        matches_dict = {}

        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            try:
                json_obj = json.loads(line)

                # We're only interested in match objects
                if json_obj.get("type") == "match":
                    data = json_obj.get("data", {})
                    file_path = data.get("path", {}).get("text", "")
                    line_number = data.get("line_number")
                    line_content = data.get("lines", {}).get("text", "").rstrip('\n')

                    if file_path and line_number:
                        if file_path not in matches_dict:
                            matches_dict[file_path] = []
                        matches_dict[file_path].append((line_number, line_content))

            except json.JSONDecodeError:
                continue

        return matches_dict

    except FileNotFoundError:
        print("Error: ripgrep (rg) is not installed or not in PATH")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def search_with_ripgrep_simple(pattern: str, path: str = ".") -> Dict[str, List[Tuple[int, str]]]:
    """
    Alternative implementation using simpler ripgrep output (non-JSON).

    Args:
        pattern: The pattern to search for
        path: The directory to search in (default: current directory)

    Returns:
        Dictionary with filepaths as keys and list of (line_no, content) tuples as values
    """
    # Run ripgrep with simpler output
    # -n: Show line numbers
    # --no-heading: Don't group matches by file
    cmd = ["rg", "-n", "--no-heading", pattern, path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode not in [0, 1]:
            print(f"Error running ripgrep: {result.stderr}")
            return {}

        matches_dict = {}

        # Parse the output (format: filepath:line_number:content)
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            # Split only on the first two colons to handle colons in content
            parts = line.split(':', 2)
            if len(parts) >= 3:
                file_path = parts[0]
                try:
                    line_number = int(parts[1])
                    line_content = parts[2]

                    if file_path not in matches_dict:
                        matches_dict[file_path] = []
                    matches_dict[file_path].append((line_number, line_content))
                except ValueError:
                    continue

        return matches_dict

    except FileNotFoundError:
        print("Error: ripgrep (rg) is not installed or not in PATH")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}


def main():
    """Main function to demonstrate usage."""
    # Search for "sorter" in the current repository
    pattern = "sorter"

    print(f"Searching for '{pattern}' in the repository...")
    print("=" * 60)

    # Use the JSON-based approach
    results = search_with_ripgrep(pattern)

    if not results:
        print(f"No occurrences of '{pattern}' found.")
    else:
        print(f"Found occurrences in {len(results)} files:\n")

        for filepath, occurrences in results.items():
            print(f"\nFile: {filepath}")
            print(f"  Found {len(occurrences)} occurrence(s):")
            for line_no, content in occurrences:
                # Truncate long lines for display
                display_content = content[:100] + "..." if len(content) > 100 else content
                print(f"    Line {line_no}: {display_content}")

        print("\n" + "=" * 60)
        print("Results as dictionary:")
        print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    results_dict = main()