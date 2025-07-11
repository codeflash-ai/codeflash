def exponentiation(base, exponent):
    result = 1
    # Fast exponentiation by squaring (handles positive exponents as before)
    while exponent > 0:
        if exponent & 1:
            result *= base
        base *= base
        exponent >>= 1
    return result
