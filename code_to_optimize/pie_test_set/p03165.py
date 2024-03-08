def problem_p03165():
    import sys

    def ii():
        return int(sys.stdin.readline())

    def mi():
        return list(map(int, sys.stdin.readline().split()))

    def li():
        return list(map(int, sys.stdin.readline().split()))

    def li2(N):
        return [list(map(int, sys.stdin.readline().split())) for i in range(N)]

    def dp2(ini, i, j):
        return [[ini] * i for i2 in range(j)]

    def dp3(ini, i, j, k):
        return [[[ini] * i for i2 in range(j)] for i3 in range(k)]

    # import bisect #bisect.bisect_left(B, a)

    # from collections import defaultdict #d = defaultdict(int) d[key] += value

    # N = ii()

    # ss = [sys.stdin.readline().rstrip() for i in range(2*N)]

    # for i in range(N):

    S = eval(input())

    T = eval(input())

    # S = ss[2*i]

    # T = ss[2*i+1]

    ls, lt = len(S), len(T)

    dp = dp2(0, ls + 1, lt + 1)

    # if S[0] == T[0]:

    # dp[1][1] = 1

    for i in range(lt + 1):

        for j in range(ls + 1):

            if (i < lt and j < ls) and S[j] == T[i]:

                dp[i + 1][j + 1] = dp[i][j] + 1

            if j < ls:

                dp[i][j + 1] = max(dp[i][j], dp[i][j + 1])

            if i < lt:

                dp[i + 1][j] = max(dp[i][j], dp[i + 1][j])

    if not dp[lt][ls]:

        dp[lt][ls] = max(dp[lt - 1][ls], dp[lt][ls - 1])

    # print(dp[lt][ls])

    i, j = lt, ls

    ans = ""

    while dp[i][j] != 0:

        if dp[i][j] == dp[i - 1][j]:

            i -= 1

        elif dp[i][j] == dp[i][j - 1]:

            j -= 1

        else:

            ans += S[j - 1]

            i -= 1

            j -= 1

    print((ans[::-1]))


problem_p03165()
