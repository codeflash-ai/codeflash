def problem_p00016():
    import math

    x = y = 0

    theta = 90

    while True:

        d, a = list(map(int, input().split(",")))

        if d == a == 0:

            break

        x += d * math.cos(theta * math.pi / 180)

        y += d * math.sin(theta * math.pi / 180)

        theta -= a

    print((int(x)))

    print((int(y)))


problem_p00016()
