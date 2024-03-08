def problem_p00496():
    def solve():

        N, T, S = list(map(int, input().split()))

        a = [tuple(map(int, input().split())) for _ in [0] * N]

        dp = {0: 0}

        for fun, time in a:

            for _t, _f in list(dp.copy().items()):

                new_time = _t + time

                new_fun = fun + _f

                if _t < S < new_time:

                    new_time = S + time

                if new_time <= T and (new_time not in dp or new_fun > dp[new_time]):

                    dp[new_time] = new_fun

        print((max(dp.values())))

    if __name__ == "__main__":

        solve()


problem_p00496()
