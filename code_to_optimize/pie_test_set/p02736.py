def problem_p02736():
    n = int(eval(input()))

    s = eval(input())

    a = [int(s[i]) - 1 for i in range(n)]

    div = [0] * (n + 1)

    for i in range(1, n + 1):

        div[i] += div[i - 1]

        x = i

        while not x & 1:

            x >>= 1

            div[i] += 1

    XOR = 0

    for i in range(n):

        if div[n - 1] <= div[i] + div[n - 1 - i]:

            XOR ^= a[i]

    if XOR & 1:

        print((1))

        exit(0)

    for i in a:

        if i == 1:

            print((0))

            exit(0)

    XOR = 0

    for i in range(n):

        if div[n - 1] <= div[i] + div[n - 1 - i]:

            XOR ^= a[i] >> 1

    if XOR & 1:

        print((2))

        exit(0)

    print((0))


problem_p02736()
