def problem_p02846():
    s, t, a, b, x, y = list(map(int, open(0).read().split()))

    if a * s + b * t == x * s + y * t:

        print("infinity")

        exit()

    if a < x:
        a, b, x, y = x, y, a, b

    if a * s + b * t > x * s + y * t:

        print((0))

        exit()

    n = (a * s - x * s) / (x * s + y * t - (a * s + b * t))

    m = int(n)

    print((m * 2 + (n > m)))


problem_p02846()
