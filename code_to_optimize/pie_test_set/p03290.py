def problem_p03290():
    import sys

    import numpy as np

    I = np.array(sys.stdin.read().split(), dtype=np.int64)

    n, G = I[:2]

    G //= 100

    p, c = I[2:].reshape(-1, 2).T

    c //= 100

    def main():

        v = np.arange(1, n + 1)

        t = v * p + c

        combs = np.arange(2**n - 1)[:, None] >> np.arange(n) & 1

        res = p.sum()

        for comb in combs:

            s = t[comb == 1].sum()

            cnt = p[comb == 1].sum()

            if s - G < 0:

                m = np.amax(v[comb == 0])

                tmp = (G - s + m - 1) // m

                if tmp < p[m - 1]:

                    cnt += tmp

                else:

                    continue

            res = min(res, cnt)

        return res

    if __name__ == "__main__":

        ans = main()

        print(ans)


problem_p03290()
