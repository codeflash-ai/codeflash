def problem_p00767():
    l = []

    for i in range(1, 151):

        for j in range(1, 151):

            if i == j or i > j:
                continue

            l += [[i**2 + j**2, i, j]]

    l = sorted(l)

    while True:

        h, w = map(int, input().split())

        if w == h == 0:
            break

        i = l.index([w**2 + h**2, h, w]) + 1

        print(*l[i][1:], sep=" ")


problem_p00767()
