def problem_p02969(input_data):
    import sys

    input = sys.stdin.readline

    import collections
    import math

    def I():
        return int(eval(input_data))

    def MI():
        return list(map(int, input_data.split()))

    def LI():
        return list(map(int, input_data.split()))

    r = I()

    return 3 * r * r
