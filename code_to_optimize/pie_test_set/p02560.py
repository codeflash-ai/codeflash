def problem_p02560():
    #!/usr/bin/env python3

    import sys

    sys.setrecursionlimit(10**6)

    INF = 10**9 + 1  # sys.maxsize # float("inf")

    MOD = 10**9 + 7

    def floor_sum(n, m, a, b):

        ret = 0

        if a >= m:

            ret += (n - 1) * n * (a // m) // 2

            a %= m

        if b >= m:

            ret += n * (b // m)

            b %= m

        y_max = (a * n + b) // m

        x_max = y_max * m - b

        if y_max == 0:

            return ret

        ret += (n - (x_max + a - 1) // a) * y_max

        ret += floor_sum(y_max, a, m, (a - x_max % a) % a)

        return ret

    def debug(*x):

        print(*x, file=sys.stderr)

    def solve(SOLVE_PARAMS):

        pass

    def main():

        # parse input

        T = int(input())

        for _t in range(T):

            N, M, A, B = map(int, input().split())

            print(floor_sum(N, M, A, B))

    # tests

    T1 = """
    
    5
    
    4 10 6 3
    
    6 5 4 3
    
    1 1 0 0
    
    31415 92653 58979 32384
    
    1000000000 1000000000 999999999 999999999
    
    """

    TEST_T1 = """
    
    >>> as_input(T1)
    
    >>> main()
    
    3
    
    13
    
    0
    
    314095480
    
    499999999500000000
    
    """

    def _test():

        import doctest

        doctest.testmod()

        g = globals()

        for k in sorted(g):

            if k.startswith("TEST_"):

                doctest.run_docstring_examples(g[k], g, name=k)

    def as_input(s):

        "use in test, use given string as input file"

        import io

        f = io.StringIO(s.strip())

        g = globals()

        g["input"] = lambda: bytes(f.readline(), "ascii")

        g["read"] = lambda: bytes(f.read(), "ascii")

    input = sys.stdin.buffer.readline

    read = sys.stdin.buffer.read

    if sys.argv[-1] == "-t":

        print("testing")

        _test()

        sys.exit()

    main()


problem_p02560()
