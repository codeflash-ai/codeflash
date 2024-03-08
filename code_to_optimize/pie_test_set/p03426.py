def problem_p03426():
    h, w, d = list(map(int, input().split()))

    a = [list(map(int, input().split())) for i in range(h)]

    def find_element(k):

        global h, w, a

        for i in range(h):

            for j in range(w):

                if a[i][j] == k:

                    return (i, j)

    def make_distance(S, T):

        return abs(S[0] - T[0]) + abs(S[1] - T[1])

    x1 = []

    x2 = []

    for i in range(1, d + 1):

        l = (h * w - i) // d

        x1.append([0] * (l + 1))

        x2.append([[0, 0] for i in range(l + 1)])

    for i in range(h):

        for j in range(w):

            p = a[i][j] % d

            if p == 0:

                p = d - 1

            else:

                p -= 1

            l = (a[i][j] - (p + 1)) // d

            x2[p][l] = [i, j]

    for i in range(1, d + 1):

        l = len(x1[i - 1])

        for j in range(l - 1):

            x1[i - 1][j + 1] = x1[i - 1][j] + make_distance(x2[i - 1][j], x2[i - 1][j + 1])

    q = int(eval(input()))

    for i in range(q):

        l, r = list(map(int, input().split()))

        p = l % d

        if p == 0:

            v1 = l // d - 1

            v2 = r // d - 1

            print((x1[d - 1][v2] - x1[d - 1][v1]))

        else:

            v1 = l // d

            v2 = r // d

            print((x1[p - 1][v2] - x1[p - 1][v1]))

    # print(x)


problem_p03426()
