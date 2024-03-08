def problem_p00159():
    import math

    while 1:

        n = eval(input())

        if n == 0:
            break

        x = []

        for _ in [0] * n:

            i, h, w = list(map(int, input().split()))

            x.append([i, abs(w / math.pow((h / 100.0), 2) - 22)])

        x = sorted(x, key=lambda a: (a[1], a[0]))

        print(x[0][0])


problem_p00159()
