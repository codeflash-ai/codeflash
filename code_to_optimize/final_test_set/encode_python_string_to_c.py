def _encodePythonStringToC(value):
    """Encode a string, so that it gives a C string literal.

    This doesn't handle limits.
    """
    assert type(value) is bytes, type(value)
    # String builder as list for efficient concatenation
    out = []
    octal = False
    # Precompute tables for commonly checked membership
    escape_bytes = {92, 9, 13, 10, 34, 63}  # '\\', '\t', '\r', '\n', '"', '?'
    digit_bytes = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57}  # b'0123456789'

    for c in value:
        cv = c
        if cv in escape_bytes:
            out.append(r"\%03o" % cv)
            octal = True
        elif 32 <= cv <= 127:
            if octal and cv in digit_bytes:
                out.append('" "')
            out.append(chr(cv))
            octal = False
        else:
            out.append(r"\%o" % cv)
            octal = True

    # We avoid .replace('" "\\', "\\") since the logic above guarantees that
    # any needed spaces for unambiguous octal-escape will be inserted already (as '" "')
    return '"' + "".join(out) + '"'
