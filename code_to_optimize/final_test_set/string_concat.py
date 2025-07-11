def concatenate_strings(n):
    # Using list accumulation and join for faster string concatenation
    parts = []
    for i in range(n):
        parts.append(str(i))
        parts.append(", ")
    return "".join(parts)
