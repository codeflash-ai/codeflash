def gcd_recursive(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm with recursion."""
    if b == 0:
        return a
    return gcd_recursive(b, a % b)
