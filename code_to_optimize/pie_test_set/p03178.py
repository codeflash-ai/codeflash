def problem_p03178():
    K = int(eval(input()))

    D = int(eval(input()))

    MOD = 10**9 + 7

    dp = [[[0] * (int(len(str(K))) + 1) for _ in range(D)] for _ in range(2)]

    dp[0][0][0] = 1

    for digit in range(int(len(str(K)))):

        for smaller_flg in [0, 1]:

            lim = 9 if smaller_flg else int(str(K)[digit])

            for digit_num in range(lim + 1):

                new_smaller_flg = smaller_flg or digit_num < lim

                for mod_d in range(D):

                    new_mod_d = (mod_d + digit_num) % D

                    dp[new_smaller_flg][new_mod_d][digit + 1] += dp[smaller_flg][mod_d][digit]

                    dp[new_smaller_flg][new_mod_d][digit + 1] %= MOD

    print(((dp[1][0][int(len(str(K)))] + dp[0][0][int(len(str(K)))] - 1) % MOD))


problem_p03178()
