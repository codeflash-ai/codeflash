def problem_p00114():
    # 0114

    def gcd(a, b):

        if b == 0:
            return a

        return gcd(b, a % b)

    def lcm(a, b):

        return a * b / gcd(a, b)

    while True:

        try:

            x, y, z = 1, 1, 1

            c1, c2, c3 = 0, 0, 0

            a1, m1, a2, m2, a3, m3 = list(map(int, input().split()))

            if sum([a1, m1, a2, m2, a3, m3]) == 0:

                break

            while True:

                x = a1 * x % m1

                c1 += 1

                if x == 1:

                    break

            while True:

                y = a2 * y % m2

                c2 += 1

                if y == 1:

                    break

            while True:

                z = a3 * z % m3

                c3 += 1

                if z == 1:

                    break

            print((int(lcm(c1, lcm(c2, c3)))))

        except EOFError:

            break


problem_p00114()
