def problem_p03131(input_data):
    import sys

    sys.setrecursionlimit(10**7)

    f_inf = float("inf")

    mod = 10**9 + 7

    def resolve():

        k, a, b = list(map(int, input_data.split()))

        if b - a < 2:

            return k + 1

        else:

            n = k - a + 1

            res = a + n // 2 * (b - a)

            res += 1 if n % 2 != 0 else 0

            return res

    if __name__ == "__main__":

        resolve()
