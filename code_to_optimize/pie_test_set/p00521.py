def problem_p00521():
    n, m = list(map(int, input().split()))

    ss = [eval(input()) for i in range(n)]

    a1, a2 = eval(input())

    a3, a4 = eval(input())

    dic = {"J": 0, "O": 1, "I": 2}

    n1, n2, n3, n4 = dic[a1], dic[a2], dic[a3], dic[a4]

    fix = [[[0] * m for i in range(n)] for j in range(3)]

    ans = 0

    for x in range(n - 1):

        for y in range(m - 1):

            b1, b2, b3, b4 = ss[x][y], ss[x][y + 1], ss[x + 1][y], ss[x + 1][y + 1]

            if a1 == b1 and a2 == b2 and a3 == b3 and a4 == b4:

                ans += 1

                for i in range(3):

                    if i != n1:

                        fix[i][x][y] -= 1

                for i in range(3):

                    if i != n2:

                        fix[i][x][y + 1] -= 1

                for i in range(3):

                    if i != n3:

                        fix[i][x + 1][y] -= 1

                for i in range(3):

                    if i != n4:

                        fix[i][x + 1][y + 1] -= 1

            elif not (a1 == b1) and a2 == b2 and a3 == b3 and a4 == b4:

                fix[n1][x][y] += 1

            elif a1 == b1 and not (a2 == b2) and a3 == b3 and a4 == b4:

                fix[n2][x][y + 1] += 1

            elif a1 == b1 and a2 == b2 and not (a3 == b3) and a4 == b4:

                fix[n3][x + 1][y] += 1

            elif a1 == b1 and a2 == b2 and a3 == b3 and not (a4 == b4):

                fix[n4][x + 1][y + 1] += 1

    rec = 0

    for i in range(3):

        for x in range(n):

            for y in range(m):

                if fix[i][x][y] > rec:

                    rec = fix[i][x][y]

    print((ans + rec))


problem_p00521()
