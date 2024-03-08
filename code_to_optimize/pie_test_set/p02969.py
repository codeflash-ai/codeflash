def problem_p02969():
    import sys

    input = sys.stdin.readline

    import math

    import collections

    def I():
        return int(eval(input()))

    def MI():
        return list(map(int, input().split()))

    def LI():
        return list(map(int, input().split()))

    r = I()

    print((3 * r * r))


problem_p02969()
