def problem_p03387():
    a, b, c = list(map(int, input().split()))

    ans = 0

    while not (a == b == c):

        a, b, c = sorted([a, b, c])

        if a + 2 <= c:

            a += 2

        else:

            a += 1

            b += 1

        ans += 1

    print(ans)


problem_p03387()
