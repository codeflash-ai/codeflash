def problem_p03848():
    import sys

    from numpy import *

    n, *a = list(map(int, sys.stdin.read().split()))

    a = int_(bincount(a, [1] * n, n))

    print(
        (
            pow(2, n // 2, 10**9 + 7)
            if not a[0] > (n & 1) and all([x == 2 or x == 0 for x in a[1:]])
            else 0
        )
    )


problem_p03848()
