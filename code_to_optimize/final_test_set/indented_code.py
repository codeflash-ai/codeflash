def indentedCode(codes, count):
    """Indent code, used for generating test codes."""
    indent = " " * count
    return "\n".join([(indent + line) if line else "" for line in codes])
