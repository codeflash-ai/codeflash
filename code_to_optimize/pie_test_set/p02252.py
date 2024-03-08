def problem_p02252():

    class Item:

        def __init__(self, v, w):

            self.v = v

            self.w = w

            self.r = v / w

        def __str__(self):

            s = "v=" + str(self.v)

            s += ",w=" + str(self.w)

            s += ",r=" + str(self.r)

            return s

    n, w = list(map(int, input().split()))

    items = []

    for i in range(n):

        vi, wi = list(map(int, input().split()))

        items.append(Item(vi, wi))

    items.sort(key=lambda x: x.r, reverse=True)

    total = 0

    remain = w

    for i in range(n):

        v = items[i].v

        w = items[i].w

        r = items[i].r

        if w <= remain:

            total += v

            remain -= w

        else:

            total += r * remain

            break

    print(total)


problem_p02252()
