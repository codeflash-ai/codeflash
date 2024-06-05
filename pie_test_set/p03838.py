def problem_p03838(input_data):
    import math

    def abs(x):

        return math.fabs(x)

    a, b = list(map(int, input_data.split()))

    c = int(abs(b) - abs(a))

    return int(abs(c)) + (c * a < 0) + (c * b < 0) if c != 0 else 1
