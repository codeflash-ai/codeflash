def problem_p00227():
    import sys

    f = sys.stdin

    while True:

        n, m = list(map(int, f.readline().split()))

        if n == 0:

            break

        vegitables = sorted(map(int, f.readline().split()), reverse=True)

        discount = sum(vi for i, vi in enumerate(vegitables) if (i + 1) % m == 0)

        print((sum(vegitables) - discount))


problem_p00227()
