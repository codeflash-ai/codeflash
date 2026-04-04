"""Test that js_project_root is recalculated per function, not cached."""

from pathlib import Path

from codeflash.languages.javascript.test_runner import find_node_project_root


def test_find_node_project_root_returns_different_roots_for_different_files(tmp_path: Path) -> None:
    """Test that find_node_project_root returns the correct root for each file."""
    # Create main project structure
    main_project = (tmp_path / "project").resolve()
    main_project.mkdir()
    (main_project / "package.json").write_text("{}", encoding="utf-8")
    (main_project / "src").mkdir()
    main_file = (main_project / "src" / "main.ts").resolve()
    main_file.write_text("// main file", encoding="utf-8")

    # Create extension subdirectory with its own package.json
    extension_dir = (main_project / "extensions" / "discord").resolve()
    extension_dir.mkdir(parents=True)
    (extension_dir / "package.json").write_text("{}", encoding="utf-8")
    (extension_dir / "src").mkdir()
    extension_file = (extension_dir / "src" / "accounts.ts").resolve()
    extension_file.write_text("// extension file", encoding="utf-8")

    # Extension file should return extension directory
    result1 = find_node_project_root(extension_file)
    assert result1 == extension_dir, f"Expected {extension_dir}, got {result1}"

    # Main file should return main project directory
    result2 = find_node_project_root(main_file)
    assert result2 == main_project, f"Expected {main_project}, got {result2}"

    # Calling again with extension file should still return extension dir
    result3 = find_node_project_root(extension_file)
    assert result3 == extension_dir, f"Expected {extension_dir}, got {result3}"


def test_js_project_root_recalculated_per_function(tmp_path: Path) -> None:
    """Each function in a monorepo should resolve to its own nearest package.json root."""
    # Create main project
    main_project = (tmp_path / "project").resolve()
    main_project.mkdir()
    (main_project / "package.json").write_text('{"name": "main"}', encoding="utf-8")
    (main_project / "src").mkdir()

    # Create extension with its own package.json
    extension_dir = (main_project / "extensions" / "discord").resolve()
    extension_dir.mkdir(parents=True)
    (extension_dir / "package.json").write_text('{"name": "discord-extension"}', encoding="utf-8")
    (extension_dir / "src").mkdir()

    extension_file = (extension_dir / "src" / "accounts.ts").resolve()
    extension_file.write_text("export function foo() {}", encoding="utf-8")

    main_file = (main_project / "src" / "commands.ts").resolve()
    main_file.write_text("export function bar() {}", encoding="utf-8")

    js_project_root_1 = find_node_project_root(extension_file)
    assert js_project_root_1 == extension_dir

    js_project_root_2 = find_node_project_root(main_file)
    assert js_project_root_2 == main_project, (
        f"Expected {main_project}, got {js_project_root_2}. "
        f"Happens when js_project_root is not recalculated per function."
    )
