def problem_p02588():
    def five(x):

        cnt1 = 0

        while x % 5 == 0:

            cnt1 += 1

            x = x // 5

        cnt2 = 0

        while x % 2 == 0:

            cnt2 += 1

            x = x // 2

        return (min2(cnt1, 18), min2(cnt2, 18))

    def min2(x, y):

        return x if x < y else y

    import sys

    input = sys.stdin.readline

    N = int(eval(input()))

    F = []

    for i in range(N):

        A = input().rstrip()

        if "." in A:

            a, b = A.split(".")

            n = int(a + b) * 10 ** (9 - len(b))

        else:

            n = int(A) * 10**9

        F.append(five(n))

    F.sort()

    j = 0

    temp = 0

    dp = [0] * 19

    k = 0

    for i in range(N - 1):

        if N - 1 - j < i:

            dp[min2(F[i][1], 18)] -= 1

        while N - 1 - j > i and F[i][0] + F[N - 1 - j][0] >= 18:

            dp[min2(F[N - 1 - j][1], 18)] += 1

            j += 1

        k = sum(dp[: 18 - F[i][1]])

        temp += min2(j, N - i - 1) - k

    print(temp)


problem_p02588()
