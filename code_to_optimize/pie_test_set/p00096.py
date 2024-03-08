def problem_p00096():
    import sys

    a = [0] * 2001

    b = [0] * 4001

    n = 1001

    for i in range(n):

        for j in range(n):

            a[i + j] += 1

    n = 2001

    for i in range(n):

        for j in range(n):

            b[i + j] += a[i] * a[j]

    for i in sys.stdin:

        print(b[int(i)])


problem_p00096()
