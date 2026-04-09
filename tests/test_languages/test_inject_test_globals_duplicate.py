"""Test for inject_test_globals duplicate import bug.

This test reproduces the bug where AI-generated tests already have vitest imports,
but inject_test_globals() adds them again because the string-based check doesn't
catch semantic duplicates with different identifier orders.
"""

import pytest
from codeflash.languages.javascript.edit_tests import inject_test_globals
from codeflash.models.models import GeneratedTests, GeneratedTestsList
from pathlib import Path


def test_inject_test_globals_skips_existing_vitest_imports() -> None:
    """Test that inject_test_globals skips injection when vitest import already exists."""
    # AI service generated this test with vitest imports already present
    # (note: different order and identifiers than what inject_test_globals would add)
    ai_generated_test = """// vitest imports (REQUIRED for vitest - globals are NOT enabled by default)
import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
// function import
import { isWindowsDrivePath } from './infra/archive-path';

// unit tests
describe('isWindowsDrivePath', () => {
    test('should return true for Windows drive paths', () => {
        expect(isWindowsDrivePath('C:\\\\')).toBe(true);
    });
});
"""

    generated_tests = GeneratedTestsList(
        generated_tests=[
            GeneratedTests(
                generated_original_test_source=ai_generated_test,
                instrumented_behavior_test_source=ai_generated_test,
                instrumented_perf_test_source=ai_generated_test,
                behavior_file_path=Path("/tmp/test_isWindowsDrivePath.test.ts"),
                perf_file_path=Path("/tmp/test_isWindowsDrivePath_perf.test.ts"),
            )
        ]
    )

    # Call inject_test_globals for vitest + esm (this is what the CLI does)
    result = inject_test_globals(generated_tests, test_framework="vitest", module_system="esm")

    # Check that the import was NOT duplicated
    result_source = result.generated_tests[0].generated_original_test_source

    # Count how many times "from 'vitest'" appears
    import_count = result_source.count("from 'vitest'")

    # Should be exactly 1 import, not 2
    assert import_count == 1, (
        f"Expected exactly 1 vitest import, but found {import_count}. "
        f"inject_test_globals() added a duplicate import when one already existed.\n"
        f"Result:\n{result_source[:500]}"
    )

    # Also verify that we have the expected number of import statements
    # Count actual import statements, not comments containing the word "import"
    import_lines = [line for line in result_source.split('\n') if line.strip().startswith('import ')]
    assert len(import_lines) == 2, f"Should have 2 import statements (vitest + function), found {len(import_lines)}: {import_lines}"


def test_inject_test_globals_adds_import_when_missing() -> None:
    """Test that inject_test_globals DOES add import when it's truly missing."""
    # Test without any vitest imports
    test_without_imports = """// function import
import { isWindowsDrivePath } from './infra/archive-path';

describe('isWindowsDrivePath', () => {
    test('should return true', () => {
        expect(isWindowsDrivePath('C:\\\\')).toBe(true);
    });
});
"""

    generated_tests = GeneratedTestsList(
        generated_tests=[
            GeneratedTests(
                generated_original_test_source=test_without_imports,
                instrumented_behavior_test_source=test_without_imports,
                instrumented_perf_test_source=test_without_imports,
                behavior_file_path=Path("/tmp/test.test.ts"),
                perf_file_path=Path("/tmp/test_perf.test.ts"),
            )
        ]
    )

    result = inject_test_globals(generated_tests, test_framework="vitest", module_system="esm")
    result_source = result.generated_tests[0].generated_original_test_source

    # Should have exactly 1 vitest import (the one we added)
    import_count = result_source.count("from 'vitest'")
    assert import_count == 1, f"Expected vitest import to be added, found {import_count}"

    # Should be at the beginning of the file
    assert result_source.startswith("import { vi, describe, it, expect"), (
        "Vitest import should be added at the beginning"
    )
