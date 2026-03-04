from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.code_utils.config_java_validation import infer_java_module_root, validate_java_module_resolution

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateJavaModuleResolution:
    def test_valid_maven_project(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        source_file = pkg_dir / "Foo.java"
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, src_root)
        assert valid is True
        assert error == ""

    def test_source_does_not_exist(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        source_file = project_root / "src" / "main" / "java" / "Missing.java"

        valid, error = validate_java_module_resolution(source_file, project_root, project_root)
        assert valid is False
        assert "does not exist" in error

    def test_source_outside_project_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        outside_file = tmp_path / "outside" / "Foo.java"
        outside_file.parent.mkdir(parents=True)
        outside_file.write_text("public class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(outside_file, project_root, project_root)
        assert valid is False
        assert "outside the project root" in error

    def test_no_build_config(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        source_file = project_root / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, project_root)
        assert valid is False
        assert "No build configuration" in error

    def test_source_outside_module_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        module_root = project_root / "src" / "main" / "java"
        module_root.mkdir(parents=True)
        # Source file is in the project root, not in module root
        source_file = project_root / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, module_root)
        assert valid is False
        assert "outside the module root" in error

    def test_package_declaration_mismatch(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        wrong_dir = src_root / "com" / "bar"
        wrong_dir.mkdir(parents=True)
        source_file = wrong_dir / "Foo.java"
        source_file.write_text("package com.foo;\npublic class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, src_root)
        assert valid is False
        assert "does not match directory structure" in error

    def test_no_package_declaration(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        src_root.mkdir(parents=True)
        source_file = src_root / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, src_root)
        assert valid is True
        assert error == ""

    def test_gradle_project(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "build.gradle").write_text("apply plugin: 'java'", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        pkg_dir = src_root / "com" / "example"
        pkg_dir.mkdir(parents=True)
        source_file = pkg_dir / "Foo.java"
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        valid, error = validate_java_module_resolution(source_file, project_root, src_root)
        assert valid is True
        assert error == ""


class TestInferJavaModuleRoot:
    def test_infers_standard_maven_layout(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        src_root.mkdir(parents=True)
        source_file = src_root / "com" / "example" / "Foo.java"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        result = infer_java_module_root(source_file, project_root)
        assert result.resolve() == src_root.resolve()

    def test_falls_back_to_project_root(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        # No src/main/java, no alternative source dirs with .java files
        source_file = project_root / "Foo.java"
        source_file.write_text("public class Foo {}", encoding="utf-8")

        result = infer_java_module_root(source_file, project_root)
        assert result.resolve() == project_root.resolve()

    def test_detects_project_root_from_pom(self, tmp_path: Path) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
        src_root = project_root / "src" / "main" / "java"
        src_root.mkdir(parents=True)
        source_file = src_root / "com" / "example" / "Foo.java"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("package com.example;\npublic class Foo {}", encoding="utf-8")

        # Don't pass project_root — let it detect from pom.xml
        result = infer_java_module_root(source_file, project_root=None)
        assert result.resolve() == src_root.resolve()
