def exponentiation(base, exponent):
    result = 1
    for _ in range(exponent):
        result *= base
    return result
