def problem_p00458():
    def solve():

        while True:

            m, n = int(eval(input())), int(eval(input()))

            if not m:

                break

            ices = [[0] + list(map(int, input().split())) + [0] for _ in range(n)]

            ices.insert(
                0,
                [0] * (m + 2),
            )

            ices.append([0] * (m + 2))

            score = [[0] * (m + 2) for _ in range(n + 2)]

            for x in range(1, n + 1):

                for y in range(1, m + 1):

                    score[x][y] = ices[x - 1][y] + ices[x + 1][y] + ices[x][y - 1] + ices[x][y + 1]

            def search(x, y, imap, acc):

                imap[x][y] = 0

                acc += 1

                a = b = c = d = 0

                if imap[x - 1][y]:

                    a = search(x - 1, y, imap, acc)

                if imap[x + 1][y]:

                    b = search(x + 1, y, imap, acc)

                if imap[x][y - 1]:

                    c = search(x, y - 1, imap, acc)

                if imap[x][y + 1]:

                    d = search(x, y + 1, imap, acc)

                imap[x][y] = 1

                return max(acc, a, b, c, d)

            ans = 0

            for x in range(1, n + 1):

                for y in range(1, m + 1):

                    if score[x][y] in [1, 2] and ices[x][y]:

                        a = search(x, y, ices, 0)

                        if ans < a:

                            ans = a

            print(ans)

    solve()


problem_p00458()
