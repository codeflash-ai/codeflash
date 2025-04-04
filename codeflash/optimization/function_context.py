from __future__ import annotations

from jedi.api.classes import Name

from codeflash.code_utils.code_utils import get_qualified_name


def belongs_to_method(name: Name, class_name: str, method_name: str) -> bool:
    """Check if the given name belongs to the specified method."""
    return belongs_to_function(name, method_name) and belongs_to_class(name, class_name)


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function."""
    if name.name == function_name:  # Handles function definition and recursive function calls
        return False
    if name := name.parent():
        if name.type == "function":
            return name.name == function_name
    return False


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if given jedi Name is a direct child of the specified class."""
    while name := name.parent():
        if name.type == "class":
            return name.name == class_name
    return False


def belongs_to_function_qualified(name: Name, qualified_function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function, matched by qualified function name."""
    try:
        if name.full_name.startswith(name.module_name) and get_qualified_name(name.module_name, name.full_name) == qualified_function_name:
            # Handles function definition and recursive function calls
            return False
        if name := name.parent():
            if name.type == "function":
                return get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        return False
    except ValueError:
        return False
