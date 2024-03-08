def problem_p02573():
    import sys

    sys.setrecursionlimit(10**9)

    n, m = list(map(int, input().split()))

    root = [-1] * n

    def r(x):

        if root[x] < 0:

            return x

        else:

            root[x] = r(root[x])

            return root[x]

    def unite(x, y):

        x = r(x)

        y = r(y)

        if x == y:

            return

        root[x] += root[y]

        root[y] = x

    def size(x):

        x = r(x)

        return -root[x]

    for i in range(m):

        x, y = list(map(int, input().split()))

        x -= 1

        y -= 1

        unite(x, y)

    ans = 0

    for i in range(n):

        ans = max(ans, size(i))

    print(ans)


problem_p02573()
