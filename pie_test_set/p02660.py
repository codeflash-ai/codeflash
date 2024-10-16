def problem_p02660(input_data):
    def getN():

        return int(eval(input_data))

    def getNM():

        return list(map(int, input_data.split()))

    def getList():

        return list(map(int, input_data.split()))

    import copy
    import math
    import sys
    from bisect import bisect_left, bisect_right
    from collections import defaultdict, deque
    from heapq import *
    from sys import exit

    # sys.setrecursionlimit(1000000)

    INF = 10**17

    MOD = 1000000007

    from fractions import *

    def inverse(f):

        # return Fraction(f.denominator,f.numerator)

        return 1 / f

    def combmod(n, k, mod=MOD):

        ret = 1

        for i in range(n - k + 1, n + 1):

            ret *= i

            ret %= mod

        for i in range(1, k + 1):

            ret *= pow(i, mod - 2, mod)

            ret %= mod

        return ret

    def bunsu(n):

        ret = []

        for i in range(2, int(math.sqrt(n)) + 1):

            if n % i == 0:

                tmp = 0

                while True:

                    if n % i == 0:

                        tmp += 1

                        n //= i

                    else:

                        break

                ret.append((i, tmp))

        ret.append((n, 1))

        return ret

    def solve():

        n = getN()

        bun = bunsu(n)

        # return (bun)

        acc = []

        tmp = 0

        for i in range(10000):

            tmp += i

            acc.append(tmp)

        ans = 0

        for b, cnt in bun:

            if b == 1:

                continue

            ans += bisect_right(acc, cnt) - 1

        return ans

    def main():

        # n = getN()

        # for _ in range(n):

        solve()

    if __name__ == "__main__":

        solve()
