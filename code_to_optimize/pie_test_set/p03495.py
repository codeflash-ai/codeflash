def problem_p03495():
    n, k = list(map(int, input().split()))

    a = list(map(int, input().split()))

    d = {}

    for i in a:

        if i not in d:

            d[i] = 1

        else:

            d[i] += 1

    d = sorted(list(d.items()), key=lambda x: x[1], reverse=True)

    ans = 0

    if len(d) <= k:

        print((0))

    else:

        for i in range(len(d) - k):

            ans += d[-i - 1][1]

        print(ans)


problem_p03495()
