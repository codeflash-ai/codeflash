def problem_p03987():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(10**7)

    # 大きい数字から追加していく。そのときの連結成分の大きさだけ足される。

    # union findっぽく。各成分の左端を根として持つ

    N = int(eval(input()))

    A = [int(x) for x in input().split()]

    x_to_i = {x: i for i, x in enumerate(A)}

    V = set()

    root = dict()

    size = dict()

    def find_root(x):

        y = root[x]

        if x == y:

            return y

        ry = find_root(y)

        root[x] = ry

        return ry

    def merge(x, y):

        rx = find_root(x)

        ry = find_root(y)

        sx = size[rx]

        sy = size[ry]

        if sx > sy:

            rx, ry = ry, rx

            sx, sy = sy, sx

        root[rx] = ry

        size[ry] += sx

    answer = 0

    for x in range(N, 0, -1):

        i = x_to_i[x]

        V.add(i)

        size[i] = 1

        root[i] = i

        left = 0

        right = 0

        if i - 1 in V:

            left = size[find_root(i - 1)]

            merge(i - 1, i)

        if i + 1 in V:

            right = size[find_root(i + 1)]

            merge(i + 1, i)

        cnt = (left + 1) * (right + 1)

        answer += x * cnt

    print(answer)


problem_p03987()
