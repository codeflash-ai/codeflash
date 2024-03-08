def problem_p03182():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    N, M = list(map(int, input().split()))

    LRA = [[int(x) for x in row.split()] for row in sys.stdin.readlines()]

    R_to_LA = [[] for _ in range(N + 1)]

    for L, R, A in LRA:

        R_to_LA[R].append((L, A))

    INF = 10**18

    size = 1 << 18

    data = [0] * (2 * size)

    lazy = [0] * (2 * size)

    left_end = [None] * (2 * size)

    right_end = [None] * (2 * size)

    left_end[1] = 0

    right_end[1] = size

    for n in range(1, size):

        L, R = left_end[n], right_end[n]

        M = (L + R) // 2

        left_end[2 * n] = L

        right_end[2 * n] = M

        left_end[2 * n + 1] = M

        right_end[2 * n + 1] = R

    def add(L, R, x):

        q = [1]

        q1 = []

        while q:

            qq = []

            for i in q:

                iL = left_end[i]

                iR = right_end[i]

                if L <= iL and iR <= R:

                    # 完全に含まれている

                    lazy[i] += x

                elif iR <= L or R <= iL:

                    pass

                else:

                    # 部分的に交わっている

                    q1.append(i)

                    j = i << 1
                    k = j + 1

                    lazy[j] += lazy[i]
                    lazy[k] += lazy[i]

                    lazy[i] = 0

                    qq.append(j)

                    qq.append(k)

            q = qq

        for i in reversed(q1):

            j = i << 1
            k = j + 1

            data[i] = max(data[j] + lazy[j], data[k] + lazy[k])

    for R in range(1, N + 1):

        x = data[1] + lazy[1]

        add(R, R + 1, x)

        for L, A in R_to_LA[R]:

            add(L, R + 1, A)

    for n in range(size):

        lazy[2 * n] += lazy[n]

        lazy[2 * n + 1] += lazy[n]

    answer = max(lazy[size:])

    print(answer)


problem_p03182()
