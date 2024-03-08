def problem_p03735():
    import sys

    from operator import itemgetter

    inf = 1 << 30

    def solve():

        n = int(sys.stdin.readline())

        # r_max = MAX, b_min = MIN にしたとき

        r_max = b_max = 0

        r_min = b_min = inf

        p = []

        for i in range(n):

            xi, yi = list(map(int, sys.stdin.readline().split()))

            if xi > yi:

                xi, yi = yi, xi

            p.append((xi, yi))

            r_max = max(r_max, yi)

            r_min = min(r_min, yi)

            b_max = max(b_max, xi)

            b_min = min(b_min, xi)

        ans1 = (r_max - r_min) * (b_max - b_min)

        # r_max = MAX, r_min = MIN にしたとき

        ans2 = r_max - b_min

        p.sort(key=itemgetter(0))

        b_min = p[0][0]

        b_max = p[-1][0]

        y_min = inf

        dif_b = b_max - b_min

        for i in range(n - 1):

            if p[i][1] == r_max:

                break

            y_min = min(y_min, p[i][1])

            b_min = min(p[i + 1][0], y_min)

            b_max = max(b_max, p[i][1])

            dif_b = min(dif_b, b_max - b_min)

            if p[i][1] < p[i + 1][0]:

                break

        ans2 *= dif_b

        ans = min(ans1, ans2)

        print(ans)

    if __name__ == "__main__":

        solve()


problem_p03735()
