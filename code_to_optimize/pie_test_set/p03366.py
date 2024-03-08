def problem_p03366():
    from collections import Counter, defaultdict

    import math

    from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING

    from functools import lru_cache, reduce

    from itertools import combinations_with_replacement, product, combinations

    def read_int():

        return int(input())

    def read_int_n():

        return list(map(int, input().split()))

    def read_str():

        return input()

    def read_str_n():

        return list(map(str, input().split()))

    def mt(f):

        import time

        import sys

        def wrap(*args, **kwargs):

            s = time.time()

            ret = f(*args, **kwargs)

            e = time.time()

            print(e - s, "sec", file=sys.stderr)

            return ret

        return wrap

    @mt
    def slv(S, X, P):

        X_ = [[P[i], [x]] for i, x in enumerate(X)]

        while len(X_) != 1:

            if X_[0][0] >= X_[-1][0]:

                di = -1

                li = 0

            else:

                di = 0

                li = -1

            X_[li][0] += X_[di][0]

            X_[li][1] += X_[di][1]

            X_.pop(di)

        t = 0

        pos = S

        p0 = S

        p1 = S

        for o in X_[0][1]:

            if p0 < o < p1:

                continue

            t += abs(pos - o)

            if o < pos:

                p0 = o

            else:

                p1 = o

            pos = o

        return t

    def main():

        N, S = read_int_n()

        X = []

        P = []

        for _ in range(N):

            x, p = read_int_n()

            X.append(x)

            P.append(p)

        print(slv(S, X, P))

    if __name__ == "__main__":

        main()


problem_p03366()
