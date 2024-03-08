def problem_p02697():
    n, m = list(map(int, input().split()))

    if n % 2 == 1:

        ab = []

        a = n // 2

        b = n // 2 + 1

        ab.append([str(a), str(b)])

        for _ in range(m - 1):

            a -= 1

            b += 1

            ab.append([str(a), str(b)])

        for a, b in ab:

            print((a, b))

    else:

        l = m

        ab = []

        cnt = 0

        for i in range(m):

            if i % 2 == 0:

                a = i // 2 + 1

                b = a + l

            else:

                a = n - (i // 2 + 1)

                b = a - l

            ab.append([str(a), str(b)])

            l -= 1

        for a, b in ab:

            print((a, b))


problem_p02697()
