def problem_p02725():
    k, n = list(map(int, input().split()))

    a = list(map(int, input().split()))

    diff = []

    for i in range(n - 1):

        diff.append(a[i + 1] - a[i])

    diff.append(k - a[-1] + a[0])

    print((k - max(diff)))


problem_p02725()
