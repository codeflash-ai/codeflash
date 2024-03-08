def problem_p04048():
    from fractions import *

    n, x = list(map(int, input().split()))

    print((3 * (n - gcd(n, x))))


problem_p04048()
