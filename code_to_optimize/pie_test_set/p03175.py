def problem_p03175():
    # @author

    import sys

    class PIndependentSet:

        def solve(self):

            from collections import defaultdict

            import sys

            sys.setrecursionlimit(10**5 + 5)

            MOD = 10**9 + 7

            def dfs(s):

                done[s] = 1

                dp[s][0] = 1

                dp[s][1] = 1

                for x in adj[s]:

                    if done[x]:

                        continue

                    dfs(x)

                    dp[s][0] = (dp[s][0] * dp[x][1]) % MOD

                    dp[s][1] = (dp[s][1] * (dp[x][0] + dp[x][1])) % MOD

            n = int(eval(input()))

            adj = defaultdict(list)

            for i in range(n - 1):

                x, y = [int(_) for _ in input().split()]

                x -= 1

                y -= 1

                adj[x].append(y)

                adj[y].append(x)

            dp = [[0, 0] for _ in range(n)]

            done = [0] * n

            dfs(0)

            print(((dp[0][0] + dp[0][1]) % MOD))

    solver = PIndependentSet()

    input = sys.stdin.readline

    solver.solve()


problem_p03175()
