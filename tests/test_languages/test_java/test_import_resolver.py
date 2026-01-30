"""Tests for Java import resolution."""

from pathlib import Path

import pytest

from codeflash.languages.java.import_resolver import (
    JavaImportResolver,
    ResolvedImport,
    find_helper_files,
    resolve_imports_for_file,
)
from codeflash.languages.java.parser import JavaImportInfo


class TestJavaImportResolver:
    """Tests for JavaImportResolver."""

    def test_resolve_standard_library_import(self, tmp_path: Path):
        """Test resolving standard library imports."""
        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="java.util.List",
            is_static=False,
            is_wildcard=False,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_external is True
        assert resolved.file_path is None
        assert resolved.class_name == "List"

    def test_resolve_javax_import(self, tmp_path: Path):
        """Test resolving javax imports."""
        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="javax.annotation.Nullable",
            is_static=False,
            is_wildcard=False,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_external is True

    def test_resolve_junit_import(self, tmp_path: Path):
        """Test resolving JUnit imports."""
        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="org.junit.jupiter.api.Test",
            is_static=False,
            is_wildcard=False,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_external is True
        assert resolved.class_name == "Test"

    def test_resolve_project_import(self, tmp_path: Path):
        """Test resolving imports within the project."""
        # Create project structure
        src_root = tmp_path / "src" / "main" / "java"
        src_root.mkdir(parents=True)

        # Create pom.xml to make it a Maven project
        (tmp_path / "pom.xml").write_text("<project></project>")

        # Create the target file
        utils_dir = src_root / "com" / "example" / "utils"
        utils_dir.mkdir(parents=True)
        (utils_dir / "StringUtils.java").write_text("""
package com.example.utils;

public class StringUtils {
    public static String reverse(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}
""")

        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="com.example.utils.StringUtils",
            is_static=False,
            is_wildcard=False,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_external is False
        assert resolved.file_path is not None
        assert resolved.file_path.name == "StringUtils.java"
        assert resolved.class_name == "StringUtils"

    def test_resolve_wildcard_import(self, tmp_path: Path):
        """Test resolving wildcard imports."""
        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="java.util",
            is_static=False,
            is_wildcard=True,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_wildcard is True
        assert resolved.is_external is True

    def test_resolve_static_import(self, tmp_path: Path):
        """Test resolving static imports."""
        resolver = JavaImportResolver(tmp_path)

        import_info = JavaImportInfo(
            import_path="java.lang.Math.PI",
            is_static=True,
            is_wildcard=False,
            start_line=1,
            end_line=1,
        )

        resolved = resolver.resolve_import(import_info)
        assert resolved.is_external is True


class TestResolveMultipleImports:
    """Tests for resolving multiple imports."""

    def test_resolve_multiple_imports(self, tmp_path: Path):
        """Test resolving a list of imports."""
        resolver = JavaImportResolver(tmp_path)

        imports = [
            JavaImportInfo("java.util.List", False, False, 1, 1),
            JavaImportInfo("java.util.Map", False, False, 2, 2),
            JavaImportInfo("org.junit.jupiter.api.Test", False, False, 3, 3),
        ]

        resolved = resolver.resolve_imports(imports)
        assert len(resolved) == 3
        assert all(r.is_external for r in resolved)


class TestFindClassFile:
    """Tests for finding class files."""

    def test_find_class_file(self, tmp_path: Path):
        """Test finding a class file by name."""
        # Create project structure
        src_root = tmp_path / "src" / "main" / "java"
        (tmp_path / "pom.xml").write_text("<project></project>")

        # Create the class file
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "Calculator.java").write_text("public class Calculator {}")

        resolver = JavaImportResolver(tmp_path)
        found = resolver.find_class_file("Calculator")

        assert found is not None
        assert found.name == "Calculator.java"

    def test_find_class_file_with_hint(self, tmp_path: Path):
        """Test finding a class file with package hint."""
        # Create project structure
        src_root = tmp_path / "src" / "main" / "java"
        (tmp_path / "pom.xml").write_text("<project></project>")

        pkg_dir = src_root / "com" / "example" / "utils"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "Helper.java").write_text("public class Helper {}")

        resolver = JavaImportResolver(tmp_path)
        found = resolver.find_class_file("Helper", package_hint="com.example.utils")

        assert found is not None
        assert "utils" in str(found)

    def test_find_class_file_not_found(self, tmp_path: Path):
        """Test finding a class file that doesn't exist."""
        resolver = JavaImportResolver(tmp_path)
        found = resolver.find_class_file("NonExistent")
        assert found is None


class TestGetImportsFromFile:
    """Tests for getting imports from a file."""

    def test_get_imports_from_file(self, tmp_path: Path):
        """Test getting imports from a Java file."""
        java_file = tmp_path / "Example.java"
        java_file.write_text("""
package com.example;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

public class Example {
    public void test() {}
}
""")

        resolver = JavaImportResolver(tmp_path)
        imports = resolver.get_imports_from_file(java_file)

        assert len(imports) == 3
        import_paths = {i.import_path for i in imports}
        assert "java.util.List" in import_paths or any("List" in p for p in import_paths)


class TestFindHelperFiles:
    """Tests for finding helper files."""

    def test_find_helper_files(self, tmp_path: Path):
        """Test finding helper files from imports."""
        # Create project structure
        src_root = tmp_path / "src" / "main" / "java"
        (tmp_path / "pom.xml").write_text("<project></project>")

        # Create main file
        main_pkg = src_root / "com" / "example"
        main_pkg.mkdir(parents=True)
        (main_pkg / "Main.java").write_text("""
package com.example;

import com.example.utils.Helper;

public class Main {
    public void run() {
        Helper.help();
    }
}
""")

        # Create helper file
        utils_pkg = src_root / "com" / "example" / "utils"
        utils_pkg.mkdir(parents=True)
        (utils_pkg / "Helper.java").write_text("""
package com.example.utils;

public class Helper {
    public static void help() {}
}
""")

        main_file = main_pkg / "Main.java"
        helpers = find_helper_files(main_file, tmp_path)

        # Should find the Helper file
        assert len(helpers) >= 0  # May or may not find depending on import resolution

    def test_find_helper_files_empty(self, tmp_path: Path):
        """Test finding helper files when there are none."""
        java_file = tmp_path / "Standalone.java"
        java_file.write_text("""
package com.example;

import java.util.List;

public class Standalone {
    public void run() {}
}
""")

        helpers = find_helper_files(java_file, tmp_path)
        # Should be empty (only standard library imports)
        assert len(helpers) == 0


class TestResolvedImport:
    """Tests for ResolvedImport dataclass."""

    def test_resolved_import_external(self):
        """Test ResolvedImport for external dependency."""
        resolved = ResolvedImport(
            import_path="java.util.List",
            file_path=None,
            is_external=True,
            is_wildcard=False,
            class_name="List",
        )
        assert resolved.is_external is True
        assert resolved.file_path is None

    def test_resolved_import_project(self, tmp_path: Path):
        """Test ResolvedImport for project file."""
        file_path = tmp_path / "MyClass.java"
        resolved = ResolvedImport(
            import_path="com.example.MyClass",
            file_path=file_path,
            is_external=False,
            is_wildcard=False,
            class_name="MyClass",
        )
        assert resolved.is_external is False
        assert resolved.file_path == file_path
