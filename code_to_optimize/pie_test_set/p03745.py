def problem_p03745():
    N = int(eval(input()))

    src = list(map(int, input().split()))

    asc = None

    ans = 1

    for i in range(N - 1):

        d = src[i + 1] - src[i]

        if d == 0:
            continue

        if asc is None:

            asc = d > 0

        elif (asc and d < 0) or (not asc and d > 0):

            ans += 1

            asc = None

    print(ans)


problem_p03745()
