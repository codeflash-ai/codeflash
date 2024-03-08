def problem_p02452():
    # AOJ ITP2_6_B: Includes

    # Python3 2018.6.24 bal4u

    from bisect import bisect_left

    n = int(eval(input()))

    a = list(map(int, input().split()))

    a.append(1000000001)  # > 1,000,000,000

    m = int(eval(input()))

    b = list(map(int, input().split()))

    if len(b) > len(a):
        print((0))

    else:

        p, f = 0, 1

        for k in b:

            p = bisect_left(a, k, p)

            if a[p] != k:

                f = 0

                break

        print(f)


problem_p02452()
