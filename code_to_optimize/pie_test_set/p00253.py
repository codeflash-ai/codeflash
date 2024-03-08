def problem_p00253():
    from itertools import count

    def get_weed(h):

        d10 = h[1] - h[0]

        d21 = h[2] - h[1]

        d32 = h[3] - h[2]

        if d10 == d21 == d32:

            for hi, expect in zip(h, count(h[0], d10)):

                if hi != expect:

                    return hi

        d43 = h[4] - h[3]

        if d21 == d32 == d43:

            return h[0]

        elif h[2] - h[0] == d32 == d43:

            return h[1]

        elif d10 == h[3] - h[1] == d43:

            return h[2]

        elif d10 == d21 == h[4] - h[2]:

            return h[3]

    import sys

    f = sys.stdin

    while True:

        n = int(f.readline())

        if n == 0:

            break

        h = list(map(int, f.readline().split()))

        print((get_weed(h)))


problem_p00253()
