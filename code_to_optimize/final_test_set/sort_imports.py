import isort
import logging


def sort_imports(imports_sort_cmd: str, should_sort_imports: bool, path: str) -> str:
    if imports_sort_cmd.lower() == "disabled" or not should_sort_imports:
        with open(path, encoding="utf8") as f:
            code = f.read()
        return code

    try:
        # Deduplicate and sort imports
        isort.file(path)
    except Exception as e:
        logging.exception(f"Failed to sort imports with isort for {path}: {e}")

    with open(path, encoding="utf8") as f:
        new_code = f.read()
    return new_code
