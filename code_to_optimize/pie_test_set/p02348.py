def problem_p02348():
    import math

    def lazy_update(s, e, value):

        a = [(1, l + s, l + e, 1)]

        while a:

            k, s, e, depth = a.pop()

            l_end, r_end = k << (rank - depth), (k + 1) << (rank - depth)

            mid = (l_end + r_end) // 2

            if s == l_end and e == r_end:

                lazy[k] = value

            else:

                if lazy[k] is not None:

                    lazy[k << 1] = lazy[k]

                    lazy[(k << 1) + 1] = lazy[k]

                    lazy[k] = None

                if s < mid:

                    a.append((k << 1, s, min(mid, e), depth + 1))

                if e > mid:

                    a.append(((k << 1) + 1, max(mid, s), e, depth + 1))

    def get_value(i):

        i += l

        for j in range(rank, -1, -1):

            n = lazy[i >> j]

            if n is not None:

                return n

        return tree[i]

    n, q = list(map(int, input().split()))

    l = 1 << math.ceil(math.log2(n))

    tree = [2**31 - 1] * (2 * l)

    lazy = [None] * (2 * l)

    rank = int(math.log2(len(tree)))

    ans = []

    ap = ans.append

    for _ in [None] * q:

        query = list(map(int, input().split()))

        if query[0] == 0:

            lazy_update(query[1], query[2] + 1, query[3])

        else:

            ap(get_value(query[1]))

    print(("\n".join((str(n) for n in ans))))


problem_p02348()
