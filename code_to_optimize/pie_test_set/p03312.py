def problem_p03312():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    l, r = a[:2]

    m = 1

    b = [(l, r)]

    for i in range(2, n - 2):

        r += a[i]

        for j in range(m, i):

            nl, nr = l + a[j], r - a[j]

            if abs(l - r) < abs(nl - nr):

                break

            l, r = nl, nr

            m += 1

        b.append((l, r))

    l, r = a[-2:]

    m = n - 2

    c = [(l, r)]

    for i in range(2, n - 2)[::-1]:

        l += a[i]

        for j in range(i + 1, m + 1)[::-1]:

            nl, nr = l - a[j], r + a[j]

            if abs(l - r) < abs(nl - nr):

                break

            l, r = nl, nr

            m -= 1

        c.append((l, r))

    r = 10**9

    for i in range(n - 3):

        x = max(b[i] + c[-i - 1])

        y = min(b[i] + c[-i - 1])

        r = min(r, x - y)

    print(r)


problem_p03312()
