def problem_p03390():
    def getKth(x, a, k):

        if k < a:

            return k

        return k + 1

    def getRKth(x, a, k):

        if x < a:

            return x - k + 1

        if k < (x - a):

            return x - k + 1

        return x - k

    def NB(x, b):

        if x < b:

            return x

        return x + 1

    def f(x, a, b):

        half = (x + 1) / 2

        rep = 0

        for delta in range(-50, 51):

            if half + delta >= 1 and half + delta <= x:

                rep = max(rep, getKth(x, a, half + delta) * getRKth(NB(x, b), b, half + delta))

        half = b - a

        for delta in range(-50, 51):

            if half + delta >= 1 and half + delta <= x:

                rep = max(rep, getKth(x, a, half + delta) * getRKth(NB(x, b), b, half + delta))

        half = NB(x, b) - b

        for delta in range(-50, 51):

            if half + delta >= 1 and half + delta <= x:

                rep = max(rep, getKth(x, a, half + delta) * getRKth(NB(x, b), b, half + delta))

        return rep

    q = int(input())

    while q > 0:

        a, b = list(map(int, input().split()))

        if a > b:

            a, b = b, a

        lo = 1

        hi = 10**19

        res = 0

        # print f(1, a, b)

        # print f(2, a, b)

        # print f(3, a, b)

        # print getKth(2, a, 1), getRKth(NB(2, b), b, 1)

        # print getKth(2, a, 2), getRKth(NB(2, b), b, 2)

        while lo <= hi:

            mid = (lo + hi) / 2

            if f(mid, a, b) < a * b:

                lo = mid + 1

                res = mid

            else:

                hi = mid - 1

        print(res)

        q -= 1


problem_p03390()
