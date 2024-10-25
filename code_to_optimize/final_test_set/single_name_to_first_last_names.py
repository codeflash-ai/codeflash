def single_name_to_first_last_names(
    name: str,
) -> list[tuple[str, str]]:
    parts = name.upper().split()
    if len(parts) == 2:
        return [tuple(parts)]
    elif len(parts) == 3:
        a, b, c = parts
        return [(a, c), (a, f"{b} {c}"), (f"{a} {b}", c)]
    else:
        return []
