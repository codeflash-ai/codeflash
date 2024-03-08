def problem_p03510():

    #####segfunc######

    def segfunc(x, y):

        return max(x, y)

    def init(init_val):

        # set_val

        for i in range(n):

            seg[i + num - 1] = init_val[i]

        # built

        for i in range(num - 2, -1, -1):

            seg[i] = segfunc(seg[2 * i + 1], seg[2 * i + 2])

    def update(k, x):

        k += num - 1

        seg[k] = x

        while k + 1:

            k = (k - 1) // 2

            seg[k] = segfunc(seg[k * 2 + 1], seg[k * 2 + 2])

    def query(p, q):

        if q <= p:

            return ide_ele

        p += num - 1

        q += num - 2

        res = ide_ele

        while q - p > 1:

            if p & 1 == 0:

                res = segfunc(res, seg[p])

            if q & 1 == 1:

                res = segfunc(res, seg[q])

                q -= 1

            p = p // 2

            q = (q - 1) // 2

        if p == q:

            res = segfunc(res, seg[p])

        else:

            res = segfunc(segfunc(res, seg[p]), seg[q])

        return res

    ide_ele = -(10**30)

    n = int(eval(input()))

    num = 2 ** (n - 1).bit_length()

    seg = [ide_ele] * 2 * num

    x = []

    s = []

    for i in range(n):

        p, q = list(map(int, input().split()))

        x.append(p)

        s.append(q)

    r = [0]

    for i in range(n):

        r.append(r[-1] + s[i])

    b = []

    for i in range(n):

        b.append(r[i + 1] - x[i])

    init(b)

    ans = 0

    for i in range(n):

        ans = max(ans, query(i, n) - (r[i] - x[i]))

    print(ans)


problem_p03510()
