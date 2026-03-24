from __future__ import annotations


def get_qualified_name(module_name: str, full_qualified_name: str) -> str:
    if not full_qualified_name:
        msg = "full_qualified_name cannot be empty"
        raise ValueError(msg)
    if not full_qualified_name.startswith(module_name):
        msg = f"{full_qualified_name} does not start with {module_name}"
        raise ValueError(msg)
    if module_name == full_qualified_name:
        msg = f"{full_qualified_name} is the same as {module_name}"
        raise ValueError(msg)
    return full_qualified_name[len(module_name) + 1 :]
