def _encodePythonStringToC(value):
    """Encode a string, so that it gives a C string literal.

    This doesn't handle limits.
    """
    assert type(value) is bytes, type(value)

    result = ""
    octal = False

    for c in value:
        if str is bytes:
            cv = ord(c)
        else:
            cv = c

        if c in b'\\\t\r\n"?':
            result += r"\%03o" % cv

            octal = True
        elif 32 <= cv <= 127:
            if octal and c in b"0123456789":
                result += '" "'

            result += chr(cv)

            octal = False
        else:
            result += r"\%o" % cv

            octal = True

    result = result.replace('" "\\', "\\")

    return '"%s"' % result
