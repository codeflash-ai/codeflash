def problem_p03148():
    from sys import exit

    N, K = list(map(int, input().split()))

    a = {}  # {種類:美味しさ}

    b = []  # [[種類,美味しさ],...]

    for _ in range(N):

        t, d = list(map(int, input().split()))

        b.append([t, d])

        try:

            a[t].append(d)

        except KeyError:

            a[t] = [d]

    b.sort(key=lambda x: -x[1])

    f = [0] * (N + 1)

    x = 0

    point = 0

    pointsum = []

    for [t, d] in b[:K]:

        if f[t] == 0:
            x += 1

        f[t] += 1

        point += d

    maxpoint = point + x * x

    h = K - 1

    for [t, d] in b[K:]:

        if f[t] == 0:

            f[t] = 1

            x += 1

            point += d

            for i in range(h, -2, -1):

                if i == -1:

                    print(maxpoint)

                    exit()

                t = b[i][0]

                d = b[i][1]

                if f[t] > 1:

                    f[t] -= 1

                    point -= d

                    h = i - 1

                    break

            maxpoint = max(maxpoint, point + x * x)

    print(maxpoint)


problem_p03148()
