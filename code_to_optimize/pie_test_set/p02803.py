def problem_p02803():
    from collections import deque

    h, w = list(map(int, input().split()))

    a = [eval(input()) + "#" for _ in range(h)]

    b = ["#" * (w + 1)]

    r = 0

    for i in range(h):

        for j in range(w):

            if a[i][j] == ".":

                b[:-1] = list(map(list, a))

                b[i][j] = 0

                q = deque([(i, j)])

                while q:

                    i, j = q.popleft()

                    r = max(r, b[i][j])

                    d = 0

                    for i1 in range(i - 1, i + 2):

                        for j1 in range(j - d, j + 2, 2):

                            if b[i1][j1] == ".":

                                b[i1][j1] = b[i][j] + 1

                                q.append((i1, j1))

                        d ^= 1

    print(r)


problem_p02803()
