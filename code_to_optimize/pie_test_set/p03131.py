def problem_p03131():
    import sys

    sys.setrecursionlimit(10**7)

    f_inf = float("inf")

    mod = 10**9 + 7

    def resolve():

        k, a, b = list(map(int, input().split()))

        if b - a < 2:

            print((k + 1))

        else:

            n = k - a + 1

            res = a + n // 2 * (b - a)

            res += 1 if n % 2 != 0 else 0

            print(res)

    if __name__ == "__main__":

        resolve()


problem_p03131()
