def problem_p00489():
    N = int(input())

    s = [0] * N

    for _ in [0] * (N * ~-N // 2):

        a, b, c, d = map(int, input().split())

        s[a - 1] += 3 * (c > d) + (c == d)

        s[b - 1] += 3 * (d > c) + (d == c)

    b = [[] for _ in [0] * N * 3]

    for i in range(N):

        b[s[i]] += [i]

    r = 1

    for x in b[::-1]:

        for y in x:

            s[y] = r

        if x:
            r += len(x)

    print(*s, sep="\n")


problem_p00489()
