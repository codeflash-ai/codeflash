"""Test that TypeScript test files preserve ESM imports in CommonJS projects.

Regression test for Issue #15: TypeScript tests converted to CommonJS cause SyntaxError.

When the AI service generates tests for TypeScript files in CommonJS projects:
1. AI service generates ESM import syntax (correct per Issue #12 fix)
2. CLI should NOT convert these imports to CommonJS require()
3. TypeScript test runners (@swc/jest, ts-jest) expect ESM syntax in .ts files

If imports are converted to require(), Jest fails with SyntaxError when trying
to load TypeScript source files via require().

Trace ID: 024aacf1-42c9-4e06-a27b-870660035d3e
"""

from pathlib import Path

import pytest

from codeflash.languages.javascript.module_system import ensure_module_system_compatibility


class TestTypeScriptTestESMPreservation:
    """Tests for preserving ESM imports in TypeScript test files."""

    def test_typescript_test_preserves_esm_in_commonjs_project(self):
        """TypeScript test files should keep ESM imports even in CommonJS projects."""
        # TypeScript test with ESM imports (what AI service generates)
        typescript_test = """import { destroy } from '../../internal';
import sdk from '../../../sdk';

test('should work', () => {
    expect(destroy).toBeDefined();
});
"""

        # Convert to CommonJS (simulating CommonJS project)
        # For TypeScript tests, this should be a NO-OP
        result = ensure_module_system_compatibility(
            typescript_test,
            target_module_system="commonjs",
            project_root=None,
            file_path=Path("test_destroy__unit_test_0.test.ts"),  # TypeScript test file
        )

        # This test will FAIL until the fix is implemented
        # Should preserve ESM syntax for TypeScript tests
        assert "import { destroy } from" in result, "Named import should be preserved"
        assert "import sdk from" in result, "Default import should be preserved"
        assert "require(" not in result, "Should NOT convert to require() for TypeScript tests"

    def test_javascript_test_converts_esm_in_commonjs_project(self):
        """JavaScript test files should still convert ESM to CommonJS in CommonJS projects."""
        # JavaScript test with ESM imports
        javascript_test = """import { destroy } from '../../internal';
import sdk from '../../../sdk';

test('should work', () => {
    expect(destroy).toBeDefined();
});
"""

        # Convert to CommonJS (for JavaScript test, this SHOULD convert)
        # This behavior is CORRECT and should remain unchanged
        result = ensure_module_system_compatibility(
            javascript_test,
            target_module_system="commonjs",
            project_root=None,
        )

        # Should convert to CommonJS for JavaScript tests
        assert "const { destroy } = require(" in result, "Named import should convert to require"
        assert "const sdk = require(" in result, "Default import should convert to require"
        assert "import " not in result, "Should NOT have ESM imports for JavaScript tests"

    @pytest.mark.skip(reason="Test demonstrates intended behavior, but we can't distinguish source vs test files yet")
    def test_typescript_source_converts_esm_in_commonjs_project(self):
        """TypeScript SOURCE files (not tests) should still convert in CommonJS projects."""
        # This test ensures we only special-case TypeScript TEST files
        # NOTE: Currently we can't distinguish source files from test files without additional context
        # This test is skipped because it would require API changes
        typescript_source = """import { foo } from './bar';
export const result = foo();
"""

        # Convert to CommonJS (for source files, should still convert)
        result = ensure_module_system_compatibility(
            typescript_source,
            target_module_system="commonjs",
            project_root=None,
        )

        # Source files should convert normally
        assert "const { foo } = require(" in result, "Source file should convert to require"
        assert "import " not in result, "Source file should not have ESM imports"

    def test_typescript_test_with_multiple_import_styles(self):
        """Test all import styles are preserved for TypeScript tests."""
        typescript_test = """import { destroy, create } from '../../internal';
import * as utils from '../../../utils';
import sdk from '../../../sdk';
import type { Table } from '@types';

describe('tests', () => {
    test('should work', () => {
        expect(destroy).toBeDefined();
    });
});
"""

        result = ensure_module_system_compatibility(
            typescript_test,
            target_module_system="commonjs",
            project_root=None,
            file_path=Path("test.spec.ts"),  # TypeScript test file
        )

        # All import styles should be preserved for TypeScript tests
        assert "import { destroy, create } from" in result
        assert "import * as utils from" in result
        assert "import sdk from" in result
        assert "import type { Table } from" in result
        assert "require(" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
