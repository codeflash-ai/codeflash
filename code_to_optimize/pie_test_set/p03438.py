def problem_p03438():
    import numpy as np

    n = int(eval(input()))

    a = list(map(int, input().split()))

    b = list(map(int, input().split()))

    d, e = 0, 0

    for i in range(n):

        c = a[i] - b[i]

        if c > 0:

            d += c

        elif c % 2 == 0:

            e += -c

        else:

            e += -c - 1

    print(("Yes" if 2 * d <= e else "No"))


problem_p03438()
