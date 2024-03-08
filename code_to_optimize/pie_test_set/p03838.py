def problem_p03838():
    import math

    def abs(x):

        return math.fabs(x)

    a, b = list(map(int, input().split()))

    c = int(abs(b) - abs(a))

    print((int(abs(c)) + (c * a < 0) + (c * b < 0) if c != 0 else 1))


problem_p03838()
