def problem_p01438():
    from collections import defaultdict

    while 1:

        n = int(input())

        if n == 0:
            break

        L = [0] * n

        D = [0] * n

        for man in range(n):

            m, l = list(map(int, input().split()))

            L[man] = l

            t = 0

            for date in range(m):

                s, e = list(map(int, input().split()))

                for s in range(s - 6, e - 6):

                    t |= 1 << s

            D[man] = t

        dp = [defaultdict(int) for i in range(n)]

        dp[0][D[0]] = L[0]

        for i in range(1, n):

            for bit in list(dp[i - 1].keys()):

                if bit & D[i] == 0:

                    dp[i][bit | D[i]] = max(dp[i][bit | D[i]], dp[i - 1][bit] + L[i])

                dp[i][bit] = max(dp[i][bit], dp[i - 1][bit])

            dp[i][D[i]] = max(dp[i][D[i]], L[i])

        ans = max(max(dp[i].values()) for i in range(n))

        print(ans)


problem_p01438()
