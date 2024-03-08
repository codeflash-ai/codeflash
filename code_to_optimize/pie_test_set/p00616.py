def problem_p00616():
    # AOJ 1030 Cubes Without Holes

    # Python3 2018.7.6 bal4u

    import sys

    from sys import stdin

    input = stdin.readline

    # n <= 500,  2^9 = 512

    while True:

        n, h = list(map(int, input().split()))

        if n == 0:
            break

        ans = []

        for i in range(h):

            c, a, b = input().split()

            a, b = int(a) - 1, int(b) - 1

            if c == "xy":

                ans += [a + (b << 9) + (z << 18) for z in range(n)]

            elif c == "xz":

                ans += [a + (y << 9) + (b << 18) for y in range(n)]

            else:

                ans += [x + (a << 9) + (b << 18) for x in range(n)]

        print((n**3 - len(set(ans))))


problem_p00616()
