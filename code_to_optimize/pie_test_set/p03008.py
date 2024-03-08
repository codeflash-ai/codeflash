def problem_p03008():
    import sys

    input = sys.stdin.readline

    def F(N, A, B):

        # n 個のどんぐりを持っている

        # a個がbの価値に変換できる

        AB = [(x, y) for x, y in zip(A, B) if x < y]

        dp = [0] * (N + 1)

        for n in range(N + 1):

            x = n

            for a, b in AB:

                if n >= a:

                    y = dp[n - a] + b

                    if x < y:

                        x = y

            dp[n] = x

        return dp[N]

    N = int(eval(input()))

    A = [int(x) for x in input().split()]

    B = [int(x) for x in input().split()]

    answer = F(F(N, A, B), B, A)

    print(answer)


problem_p03008()
