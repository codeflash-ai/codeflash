from __future__ import annotations

from pathlib import Path


def get_java_test_file_path(
    test_dir: Path,
    function_name: str,
    iteration: int = 0,
    test_type: str = "unit",
    package_name: str | None = None,
    class_name: str | None = None,
) -> Path:
    assert test_type in {"unit", "inspired", "replay", "perf"}
    function_name_safe = function_name.replace(".", "_")
    extension = ".java"

    package_path = (package_name or "").replace(".", "/")
    java_class_name = class_name or f"{function_name_safe.title()}Test"
    # Add suffix to avoid conflicts
    if test_type == "perf":
        java_class_name = f"{java_class_name}__perfonlyinstrumented"
    elif test_type == "unit":
        java_class_name = f"{java_class_name}__perfinstrumented"
    path = test_dir / package_path / f"{java_class_name}{extension}"
    # Create package directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return get_java_test_file_path(test_dir, function_name, iteration + 1, test_type, package_name, class_name)
    return path
