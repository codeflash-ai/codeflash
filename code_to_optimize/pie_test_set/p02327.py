def problem_p02327():
    from sys import stdin

    from collections import deque

    readline = stdin.readline

    h, w = list(map(int, readline().split()))

    w += 1

    dp = [list(map(int, readline().split())) + [1] for _ in range(h)]

    for i in range(h):

        for j in range(w):

            dp[i][j] ^= 1

    for i in range(1, h):

        for j in range(w):

            if dp[i][j]:

                dp[i][j] += dp[i - 1][j]

    max_s = 0

    for i in range(h):

        stack = deque()

        for j in range(w):

            if (stack[-1][0] if stack else 0) < dp[i][j]:

                stack.append((dp[i][j], j))

            elif stack and stack[-1][0] > dp[i][j]:

                while stack and stack[-1][0] > dp[i][j]:

                    height, l = stack.pop()

                    s = height * (j - l)

                    if max_s < s:

                        max_s = s

                if dp[i][j]:

                    stack.append((dp[i][j], l))

    print(max_s)


problem_p02327()
