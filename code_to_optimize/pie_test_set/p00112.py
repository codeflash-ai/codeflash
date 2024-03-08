def problem_p00112():
    while 1:

        n = int(input())

        if n == 0:
            break

        x = sorted([int(input()) for _ in [0] * n])

        s = 0

        a = 0

        for e in x[:-1]:
            a += e
            s += a

        print(s)


problem_p00112()
