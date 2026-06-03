"""
Test for Issue #10: TypeScript files in CommonJS packages should not be detected as ESM

When a TypeScript file exists in a package without "type": "module" in package.json,
the module system should be detected as CommonJS, not ESM.
"""

import json
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.module_system import ModuleSystem, detect_module_system


def test_typescript_file_in_commonjs_package():
    """TypeScript file in package without 'type' field should be CommonJS"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json without "type" field (defaults to CommonJS)
        package_json = project_root / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-package",
            "version": "1.0.0"
        }))

        # Create TypeScript file with ESM source syntax
        ts_file = project_root / "src" / "index.ts"
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        ts_file.write_text("""
import { foo } from './foo';

export function bar() {
    return foo();
}
""")

        # Should detect as CommonJS, not ESM
        result = detect_module_system(project_root, ts_file)

        assert result == ModuleSystem.COMMONJS, (
            f"Expected CommonJS for TypeScript file in package without 'type' field, got {result}"
        )


def test_typescript_file_in_explicit_commonjs_package():
    """TypeScript file in package with 'type': 'commonjs' should be CommonJS"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with explicit "type": "commonjs"
        package_json = project_root / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-package",
            "version": "1.0.0",
            "type": "commonjs"
        }))

        # Create TypeScript file
        ts_file = project_root / "src" / "index.ts"
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        ts_file.write_text("""
import { foo } from './foo';
export function bar() { return foo(); }
""")

        # Should detect as CommonJS
        result = detect_module_system(project_root, ts_file)

        assert result == ModuleSystem.COMMONJS, (
            f"Expected CommonJS for TypeScript file in explicit CommonJS package, got {result}"
        )


def test_typescript_file_in_esm_package():
    """TypeScript file in package with 'type': 'module' should be ESM"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with "type": "module"
        package_json = project_root / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-package",
            "version": "1.0.0",
            "type": "module"
        }))

        # Create TypeScript file
        ts_file = project_root / "src" / "index.ts"
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        ts_file.write_text("""
import { foo } from './foo';
export function bar() { return foo(); }
""")

        # Should detect as ESM
        result = detect_module_system(project_root, ts_file)

        assert result == ModuleSystem.ES_MODULE, (
            f"Expected ESM for TypeScript file in ESM package, got {result}"
        )


def test_mts_file_always_esm():
    """.mts files should always be ESM regardless of package.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json without "type" field
        package_json = project_root / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-package",
            "version": "1.0.0"
        }))

        # Create .mts file (explicit ESM extension)
        mts_file = project_root / "src" / "index.mts"
        mts_file.parent.mkdir(parents=True, exist_ok=True)
        mts_file.write_text("""
import { foo } from './foo';
export function bar() { return foo(); }
""")

        # Should detect as ESM (explicit extension)
        result = detect_module_system(project_root, mts_file)

        assert result == ModuleSystem.ES_MODULE, (
            f"Expected ESM for .mts file, got {result}"
        )


def test_cts_file_always_commonjs():
    """.cts files should always be CommonJS regardless of package.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with "type": "module"
        package_json = project_root / "package.json"
        package_json.write_text(json.dumps({
            "name": "test-package",
            "version": "1.0.0",
            "type": "module"
        }))

        # Create .cts file (explicit CommonJS extension)
        cts_file = project_root / "src" / "index.cts"
        cts_file.parent.mkdir(parents=True, exist_ok=True)
        cts_file.write_text("""
import { foo } from './foo';
export function bar() { return foo(); }
""")

        # Should detect as CommonJS (explicit extension)
        result = detect_module_system(project_root, cts_file)

        assert result == ModuleSystem.COMMONJS, (
            f"Expected CommonJS for .cts file, got {result}"
        )
