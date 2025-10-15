#!/usr/bin/env python3
"""Example of using the ripgrep search script programmatically."""

import json

from ripgrep_search import search_with_ripgrep

# Search for any pattern you want
pattern = "sorter"  # Change this to any pattern you need
results = search_with_ripgrep(pattern)

# Access the results as a dictionary
print(f"Found matches in {len(results)} files")

# Iterate through the results
for filepath, occurrences in results.items():
    print(f"\n{filepath}: {len(occurrences)} matches")
    for line_no, content in occurrences[:3]:  # Show first 3 matches per file
        print(f"  Line {line_no}: {content[:80]}...")

# Save results to a JSON file if needed
with open("search_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Or filter results for specific files
python_files_only = {path: matches for path, matches in results.items() if path.endswith(".py")}

print(f"\nPython files with matches: {len(python_files_only)}")
