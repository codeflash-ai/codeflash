def problem_p04048(input_data):
    from fractions import *

    n, x = list(map(int, input_data.split()))

    return 3 * (n - gcd(n, x))
