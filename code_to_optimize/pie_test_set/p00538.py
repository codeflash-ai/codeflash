def problem_p00538():
    n = int(eval(input()))

    a = [int(eval(input())) for _ in range(n)]

    dp = [
        [a[i] if i == j else max(a[i], a[(i + 1) % n]) if (i + 1) % n == j else 0 for j in range(n)]
        for i in range(n)
    ]

    for i in range(3 if n % 2 == 0 else 2, n, 2):

        for l in range(n):

            r = (l + i) % n

            pat = []

            for x, nextl, nextr in [(l, (l + 1) % n, r), (r, l, (r + n - 1) % n)]:

                if a[nextl] > a[nextr]:

                    nextl = (nextl + 1) % n

                else:

                    nextr = (nextr + n - 1) % n

                pat.append(a[x] + dp[nextl][nextr])

            dp[l][r] = max(pat)

    print((max(dp[(i + 1) % n][i] for i in range(n))))


problem_p00538()
