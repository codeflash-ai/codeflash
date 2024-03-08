def problem_p00508():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0585



    """

    import sys

    from sys import stdin

    from collections import namedtuple

    input = stdin.readline

    def closest_part(points, n):

        # ?????¬???p324???

        if n <= 1:

            return float("inf")

        m = n // 2

        x = points[m][0]

        d = min(closest_part(points[:m], m), closest_part(points[m:], n - m))

        points.sort(key=lambda p: p[1])

        b = []

        for p in points:

            if (p[0] - x) ** 2 >= d:

                continue

            for q in b[::-1]:

                # dx = p[0] - b[-j-1][0]

                # dy = p[1] - b[-j-1][1]

                dx = p[0] - q[0]

                dy = p[1] - q[1]

                if dy**2 >= d:

                    break

                d = min(d, (dx**2 + dy**2))

            b.append(p)

        return d

    def main(args):

        n = int(eval(input()))

        points = [tuple(map(int, input().split())) for _ in range(n)]

        points.sort()  #  x????????§??????????????????

        result = closest_part(points, n)

        print(result)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00508()
