def problem_p00154():
    import sys

    def main():

        while True:

            m = eval(input())

            if m == 0:
                return 0

            dp = {}

            dp[(0, 0)] = 1

            for i in range(m):

                v, c = list(map(int, input().split()))

                for j in range(1001):

                    for k in range(c + 1):

                        key = (i + 1, j + v * k)

                        dp.setdefault(key, 0)

                        dp[key] += dp.get((i, j), 0)

            n = eval(input())

            for i in range(n):

                x = eval(input())

                print(dp.get((m, x), 0))

    if __name__ == "__main__":

        sys.exit(main())


problem_p00154()
