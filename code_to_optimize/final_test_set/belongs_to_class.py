from jedi.api.classes import Name


def belongs_to_class(name: Name, class_name: str) -> bool:
    """
    Check if the given name belongs to the specified class.
    """
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name[len(name.module_name) + 1 :]
        class_prefix: str = f"{class_name}."
        return subname.startswith(class_prefix)
    return False
