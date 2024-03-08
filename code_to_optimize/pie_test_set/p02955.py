def problem_p02955():
    from math import sqrt

    from itertools import accumulate

    def common_divisors(x):

        ret = []

        for i in range(1, int(sqrt(x)) + 1):

            if x % i == 0:

                ret.append(i)

                ret.append(x // i)

        return ret

    n, k = list(map(int, input().split()))

    a = list(map(int, input().split()))

    sm = sum(a)

    cd = common_divisors(sm)

    ans = 1

    for ecd in cd:

        r = [e % ecd for e in a]

        r.sort()

        acc = [0] + list(accumulate(r))

        for i in range(1, n + 1):

            sub = acc[i - 1]

            add = ecd * (n - i + 1) - (acc[n] - acc[i - 1])

            if sub == add:

                if sub <= k:

                    ans = max(ans, ecd)

    print(ans)


problem_p02955()
