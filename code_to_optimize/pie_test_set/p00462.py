def problem_p00462():
    # -*- coding: utf-8 -*-

    """

    http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=0539



    """

    import sys

    from sys import stdin

    from bisect import bisect_right

    input = stdin.readline

    def main(args):

        while True:

            d = int(eval(input()))  #  ??°??¶????????????

            if d == 0:

                break

            n = int(eval(input()))  #  ??¬????????????????????°

            m = int(eval(input()))  #  ??¨????????°

            cw_pos = [int(eval(input())) for _ in range(n - 1)]

            dests = [int(eval(input())) for _ in range(m)]

            cw_pos.append(0)

            cw_pos.append(d)

            cw_pos.sort()

            ccw_pos = [d - x for x in cw_pos]

            ccw_pos.sort()

            total_distance = 0

            for t in dests:

                if t == 0:

                    continue

                i = bisect_right(cw_pos, t)

                a1 = min(t - cw_pos[i - 1], cw_pos[i] - t)

                j = bisect_right(ccw_pos, d - t)

                a2 = min(d - t - ccw_pos[j - 1], ccw_pos[j] - (d - t))

                ans = min(a1, a2)

                total_distance += ans

            print(total_distance)

    if __name__ == "__main__":

        main(sys.argv[1:])


problem_p00462()
