def problem_p00472():
    n, m = list(map(int, input().split()))

    lst = [0]

    for i in range(n - 1):

        lst.append(lst[-1] + int(eval(input())))

    ans = 0

    num = 0

    for i in range(m):

        a = int(eval(input()))

        ans += abs(lst[num] - lst[num + a])

        ans %= 100000

        num += a

    print(ans)


problem_p00472()
