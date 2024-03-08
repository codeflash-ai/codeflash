def problem_p03169():
    from collections import Counter

    N = int(eval(input()))

    A = list(map(int, input().split()))

    dp = [[[0] * (N + 2) for _ in range(N + 2)] for _ in range(N + 2)]

    CA = Counter(A)

    SA = CA[1]

    SB = CA[2]

    SC = CA[3]

    for c in range(SC + 1):

        for b in range(SB + SC + 1):

            if b + c > N:

                break

            for a in range(N + 1):

                if a + b + c > N:

                    break

                S = a + b + c

                if S == 0:

                    continue

                dp[a][b][c] = N / S + a / S * dp[a - 1][b][c]
                +b / S * dp[a + 1][b - 1][c]
                +c / S * dp[a][b + 1][c - 1]

    print((dp[SA][SB][SC]))


problem_p03169()
