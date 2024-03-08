def problem_p03909():
    h, w = list(map(int, input().split()))

    s = []

    for i in range(h):

        S = list(input().split())

        s.append(S)

    for i in range(h):

        for j in range(w):

            if s[i][j] == "snuke":

                l = chr(ord("A") + j)

                print((l + str(i + 1)))

                exit()


problem_p03909()
