def problem_p03043():
    n, k = list(map(int, input().split()))

    ans = 0

    for i in range(1, n + 1):

        tmp = i

        cnt = 0

        while k > tmp:

            tmp = tmp * 2

            cnt += 1

        ans += (1 / n) * pow(1 / 2, cnt)

    print(ans)


problem_p03043()
