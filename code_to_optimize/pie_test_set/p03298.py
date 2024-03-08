def problem_p03298():
    from collections import defaultdict

    n = int(eval(input()))

    s = eval(input())

    a = s[:n]

    b = s[n:][::-1]

    d = defaultdict(int)

    e = defaultdict(int)

    def cnt(c, d):

        for bit in range(1 << n):

            x = []

            y = []

            for i in range(n):

                if bit >> i & 1:

                    x.append(c[i])

                else:

                    y.append(c[i])

            d[(str(x), str(y))] += 1

    cnt(a, d)

    cnt(b, e)

    ans = 0

    for k, v in list(d.items()):

        ans += e[k] * v

    print(ans)


problem_p03298()
