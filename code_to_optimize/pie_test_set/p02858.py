def problem_p02858(input_data):
    from fractions import gcd

    mod = 10**9 + 7

    def pow_r(x, n):

        if n == 0:  # exit case

            return 1

        if n % 2 == 0:  # standard case ① n is even

            return pow_r(x**2 % mod, n // 2) % mod

        else:  # standard case ② n is odd

            return x * pow_r(x**2 % mod, (n - 1) // 2) % mod

    h, w, t = list(map(int, input_data.split()))

    H = gcd(h, t)
    W = gcd(w, t)

    groups = H * W

    unit = (h // H, w // W)

    cnt = (2 ** (unit[0]) + 2 ** (unit[1]) - 3) % mod

    unitd = gcd(unit[0], unit[1])

    cnt += pow_r(2, unitd) % mod

    return pow_r(cnt, groups)
