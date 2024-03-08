def problem_p03552():
    N, Z, W = list(map(int, input().split()))

    (*A,) = list(map(int, input().split()))

    import sys

    sys.setrecursionlimit(10**6)

    memo = {}

    def dfs(i, turn):

        if (i, turn) in memo:

            return memo[i, turn]

        if turn == 0:

            # X

            if i == 0:

                res = abs(W - A[-1])

            else:

                res = abs(A[i - 1] - A[-1])

            for j in range(i + 1, N):

                res = max(res, dfs(j, 1))

        else:

            # Y

            if i == 0:

                res = abs(Z - A[-1])

            else:

                res = abs(A[i - 1] - A[-1])

            for j in range(i + 1, N):

                res = min(res, dfs(j, 0))

        memo[i, turn] = res

        return res

    print((dfs(0, 0)))


problem_p03552()
