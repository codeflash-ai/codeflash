def problem_p03699():
    # https://atcoder.jp/contests/abc063/tasks/arc075_a

    n = int(eval(input()))

    nums = []

    for _ in range(n):

        nums.append(int(eval(input())))

    total = sum(nums)

    dp = [[False for _ in range(total + 1)] for _ in range(n + 1)]

    dp[0][0] = True

    for i in range(n):

        num = nums[i]

        dp[i + 1][num] = True

        for j in range(total + 1):

            if dp[i][j] and num + j <= total:

                dp[i + 1][num + j] = True

            dp[i + 1][j] = dp[i][j] or dp[i + 1][j]

    for i in range(len(dp[0]))[::-1]:

        if dp[-1][i] and i % 10 != 0:

            print(i)

            break

    else:

        print((0))


problem_p03699()
