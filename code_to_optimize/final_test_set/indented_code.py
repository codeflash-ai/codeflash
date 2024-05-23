def indentedCode(codes, count):
    """Indent code, used for generating test codes."""
    return "\n".join(" " * count + line if line else "" for line in codes)
