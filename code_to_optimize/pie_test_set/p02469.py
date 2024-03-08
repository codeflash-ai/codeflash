def problem_p02469():
    #!/usr/bin/env python

    # -*- coding: utf-8 -*-

    """

    input:

    4

    1 2 3 5



    output:

    30

    """

    import sys

    import fractions

    from functools import reduce

    def gcd(x, y):

        if x < y:

            x, y = y, x

        while y > 0:

            r = x % y

            x = y

            y = r

        return x

    def lcm(a, b):

        return a * b // fractions.gcd(a, b)

    def solve(_n_list):

        return reduce(lcm, _n_list)

    if __name__ == "__main__":

        _input = sys.stdin.readlines()

        cnt = int(_input[0])

        n_list = tuple(map(int, _input[1].split()))

        print((solve(n_list)))


problem_p02469()
