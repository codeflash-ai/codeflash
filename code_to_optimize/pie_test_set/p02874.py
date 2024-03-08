def problem_p02874():
    # coding: utf-8

    import numpy as np

    def solve(*args: str) -> str:

        n = int(args[0])

        LR = [tuple(map(int, a.split())) for a in args[1:]]

        L, R = list(zip(*LR))

        ret = 0

        p, q = np.argmax(L), np.argmin(R)

        lp, rq = L[p], R[q]

        ret = max(0, 1 + rq - lp) + max(0, max(1 + r - l for l, r in LR))

        AB = [[max(0, 1 + r - lp), max(0, 1 + rq - l)] for l, r in LR]

        AB.sort(key=lambda x: (x[0], -x[1]))

        A, B = list(map(list, list(zip(*AB))))

        # for i in range(1, n):

        #     ret = max(ret, min(A[i:]) + min(B[:i]))

        b_min = 1 + 10**9

        for i in range(n - 1):

            b_min = min(b_min, B[i])

            ret = max(ret, b_min + A[i + 1])

        return str(ret)

    if __name__ == "__main__":

        print((solve(*(open(0).read().splitlines()))))


problem_p02874()
