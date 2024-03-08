def problem_p02319():
    #!/usr/bin/env python

    # -*- coding: utf-8 -*-

    """

    input:

    2 20

    5 9

    4 10



    output:

    9



    1 ??? N ??? 100

    1 ??? vi ??? 100

    1 ??? wi ??? 10,000,000

    1 ??? W ??? 1,000,000,000

    """

    import sys

    from collections import namedtuple

    MAX_N, MAX_V = 100, 100

    def solve():

        dp = [[float("inf")] * (MAX_N * MAX_V + 1) for _ in range(MAX_N + 1)]

        dp[0][0] = 0

        for i, item in enumerate(item_list):

            v, w = item.value, item.weight

            for j in range(MAX_N * MAX_V + 1):

                if j < v:

                    dp[i + 1][j] = dp[i][j]

                else:

                    dp[i + 1][j] = min(dp[i][j], dp[i][j - v] + w)

        ans = 0

        for i in range(MAX_N * MAX_V + 1):

            if dp[c_items][i] <= real_cp:

                ans = i

        return ans

    if __name__ == "__main__":

        _input = sys.stdin.readlines()

        c_items, real_cp = list(map(int, _input[0].split()))

        items = [x.split() for x in _input[1:]]

        Item = namedtuple("Item", ("value", "weight"))

        item_list = [Item(int(v), int(w)) for v, w in items]

        print((solve()))


problem_p02319()
