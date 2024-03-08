def problem_p03588():
    n = int(eval(input()))

    a = [[]] * n

    for i in range(n):

        a[i] = list(map(int, input().split()))

    a.sort()

    print((a[-1][0] + a[-1][1]))


problem_p03588()
