def problem_p02318():
    s1 = eval(input())

    s2 = eval(input())

    s1_len = len(s1)

    s2_len = len(s2)

    dp = [
        [i if j == 0 else j if i == 0 else None for j in range(s2_len + 1)]
        for i in range(s1_len + 1)
    ]

    for i in range(1, s1_len + 1):

        for j in range(1, s2_len + 1):

            n1 = dp[i - 1][j - 1] if s1[i - 1] == s2[j - 1] else dp[i - 1][j - 1] + 1

            n2 = dp[i - 1][j] + 1

            n3 = dp[i][j - 1] + 1

            dp[i][j] = min(n1, n2, n3)

    print((dp[s1_len][s2_len]))


problem_p02318()
