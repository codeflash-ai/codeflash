"""Test that TypeScript files can be transformed when Babel is installed but no TS transformer exists.

This tests the fix for the bug where projects with @babel/core but no TypeScript
transformer would fail with "experimental syntax 'flow'" error when Jest tried
to transform TypeScript files.

Related trace IDs: 26117bae-39bb-4f2f-9047-f2eb6594b7eb, and 3 others
"""
import json
import tempfile
from pathlib import Path
import pytest


def test_typescript_transform_with_babel_no_preset():
    """
    Test that _detect_typescript_transformer() returns a working config
    when @babel/core is present but no TypeScript transformer is installed.

    This fixes the bug where Jest would use babel-jest by default (when @babel/core
    is installed) but fail to transform TypeScript because Babel didn't have
    preset-typescript configured.
    """
    from codeflash.languages.javascript.test_runner import _detect_typescript_transformer

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a package.json with @babel/core but no TS transformer
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "@babel/core": "^7.22.5",
                "@babel/preset-env": "^7.22.5",
                # NO ts-jest, @swc/jest, @babel/preset-typescript, or esbuild-jest
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Test the detection
        transformer_name, transform_config = _detect_typescript_transformer(project_root)

        # With the fix: should return babel-jest with inline preset-typescript
        assert transformer_name == "babel-jest (fallback)"
        assert transform_config != ""
        assert "babel-jest" in transform_config
        assert "@babel/preset-typescript" in transform_config
        assert "\\.(ts|tsx)" in transform_config or r"\.(ts|tsx)" in transform_config


def test_generated_jest_config_handles_typescript_with_babel():
    """
    Test that the generated Jest config can transform TypeScript when only Babel is available.

    This verifies the end-to-end fix: the config should include a transform
    directive that uses babel-jest with preset-typescript.
    """
    from codeflash.languages.javascript.test_runner import _create_codeflash_jest_config

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Setup: project with @babel/core but no TS transformer
        package_json = {
            "name": "test-babel-ts",
            "devDependencies": {
                "@babel/core": "^7.22.5",
                "@babel/preset-env": "^7.22.5",
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Create a TypeScript source file (for realism, not used by config generation)
        src_dir = project_root / "src"
        src_dir.mkdir()
        (src_dir / "example.ts").write_text("""
export interface User {
    name: string;
    age: number;
}

export function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
""")

        # Generate codeflash Jest config
        config_path = _create_codeflash_jest_config(project_root, None, for_esm=False)

        assert config_path is not None
        assert config_path.exists()

        config_content = config_path.read_text()

        # The config should include a transform directive
        assert "transform:" in config_content, (
            "Generated Jest config must include a transform directive to handle "
            "TypeScript files when @babel/core is installed"
        )

        # Should use babel-jest with preset-typescript
        assert "babel-jest" in config_content
        assert "@babel/preset-typescript" in config_content

        # Should handle .ts and .tsx files (may be escaped as \. in regex)
        assert "ts" in config_content and "tsx" in config_content


def test_fallback_not_triggered_when_explicit_transformer_exists():
    """
    Test that the Babel fallback is NOT used when an explicit TypeScript transformer exists.

    When ts-jest, @swc/jest, etc. are installed, those should take precedence.
    """
    from codeflash.languages.javascript.test_runner import _detect_typescript_transformer

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Project with both @babel/core AND ts-jest
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "@babel/core": "^7.22.5",
                "ts-jest": "^29.0.0",
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        transformer_name, transform_config = _detect_typescript_transformer(project_root)

        # Should prefer ts-jest over babel fallback
        assert transformer_name == "ts-jest"
        assert "ts-jest" in transform_config
        assert "babel-jest" not in transform_config


def test_no_transformer_when_babel_not_installed():
    """
    Test that no transformer is returned when neither Babel nor TypeScript transformers exist.

    This ensures the fallback only triggers when @babel/core is present.
    """
    from codeflash.languages.javascript.test_runner import _detect_typescript_transformer

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Project with no transformers at all
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "jest": "^29.0.0",
                # NO @babel/core, NO ts-jest, etc.
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        transformer_name, transform_config = _detect_typescript_transformer(project_root)

        # Should return no transformer (Jest will use default behavior)
        assert transformer_name is None
        assert transform_config == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
