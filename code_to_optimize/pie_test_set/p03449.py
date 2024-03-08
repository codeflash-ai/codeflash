def problem_p03449():
    n = int(eval(input()))

    a1 = list(map(int, input().split()))

    a2 = list(map(int, input().split()))

    ans = [0] * n

    for i in range(n):

        ans[i] = sum(a1[: i + 1]) + sum(a2[i:])

    print((max(ans)))


problem_p03449()
