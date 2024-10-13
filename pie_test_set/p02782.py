def problem_p02782(input_data):
    import collections
    import heapq
    import sys
    from functools import cmp_to_key
    from sys import stdin

    import numpy as np

    ##  input functions for me

    def rsa(sep=""):

        if sep == "":

            return input_data.split()

        else:
            return input_data.split(sep)

    def rip(sep=""):

        if sep == "":

            return list(map(int, input_data.split()))

        else:
            return list(map(int, input_data.split(sep)))

    def ria(sep=""):

        return list(rip(sep))

    def ri():
        return int(eval(input_data))

    def rd():
        return float(eval(input_data))

    def rs():
        return eval(input_data)

    ##

    def inv(v, mod):

        return pow(v, mod - 2, mod)

    def main():

        r1, c1, r2, c2 = rip()

        MM = int(2e6 + 10)

        fact = [0] * MM

        finv = [0] * MM

        fact[0] = 1

        finv[0] = 1

        mod = int(1e9) + 7

        for i in range(1, MM):
            fact[i] = fact[i - 1] * i % mod

        finv[MM - 1] = inv(fact[MM - 1], mod)

        for i in reversed(list(range(1, MM - 1))):

            finv[i] = finv[i + 1] * (i + 1) % mod

        def sum_naive(r, c):

            # [0, r) * [0, c)

            ret = 0

            for i in range(r):

                ret += fact[i + 1 + c - 1] * finv[i + 1] * finv[c - 1] % mod

            return ret

        def sum(r, c):

            # [0, r) * [0, c)

            ret = fact[r + c] * finv[r] * finv[c] % mod

            ret += -1 + mod

            ret %= mod

            return ret

        ans = 0

        ans += sum(r2 + 1, c2 + 1)

        ans -= sum(r2 + 1, c1)

        ans -= sum(r1, c2 + 1)

        ans += sum(r1, c1)

        ans %= mod

        return ans

    if __name__ == "__main__":

        main()
