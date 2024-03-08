def problem_p02380():
    import math

    import sys

    a, b, C = [int(x) for x in sys.stdin.readline().split()]

    C = math.radians(C)

    l = a + b + math.sqrt(a * a + b * b - 2 * a * b * math.cos(C))

    h = b * math.sin(C)

    s = a * h / 2

    print(s)

    print(l)

    print(h)


problem_p02380()
