def problem_p03665():
    from operator import mul

    from functools import reduce

    n, p = list(map(int, input().split()))

    A = list(map(int, input().split()))

    even = 0

    odd = 0

    for a in A:

        if a % 2 == 0:

            even += 1

        else:

            odd += 1

    def cmb(n, r):

        r = min(n - r, r)

        if r == 0:

            return 1

        over = reduce(mul, list(range(n, n - r, -1)))

        under = reduce(mul, list(range(1, r + 1)))

        return over // under

    ans = 2**even

    cnt = 0

    if p == 0:

        for i in range(0, odd + 1, 2):

            cnt += cmb(odd, i)

    else:

        for i in range(1, odd + 1, 2):

            cnt += cmb(odd, i)

    print((ans * cnt))


problem_p03665()
