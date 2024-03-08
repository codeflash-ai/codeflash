def problem_p03420():
    n, k = [int(i) for i in input().split()]

    ans = 0

    if k == 0:

        ans = n * n

    else:

        for b in range(k + 1, n + 1):

            # number of perfect cycle

            ans += max(n // b, 0) * (b - k)

            r = n % b

            ans += max(r - k + 1, 0)

    print(ans)


problem_p03420()
