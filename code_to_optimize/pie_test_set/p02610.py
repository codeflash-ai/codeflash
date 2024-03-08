def problem_p02610():
    from heapq import *

    i = input

    for s in [0] * int(i()):

        n, x, *y = int(i()), []

        for _ in "_" * n:
            k, l, r = t = [*list(map(int, i().split()))]
            x += [t] * (l > r)
            y += [[n - k, r, l]] * (l <= r)

        for x in x, y:

            x.sort()
            h = []

            for k, l, r in x:

                s += r

                if k:
                    s += l - r
                    heappush(h, l - r)

                if len(h) > k:
                    s -= heappop(h)

        print(s)


problem_p02610()
