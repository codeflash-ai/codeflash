def problem_p02713(input_data):
    def GCD(a, b):

        return a if b == 0 else GCD(b, a % b)

    k = int(eval(input_data))

    ans = 0

    for a in range(1, k + 1):

        for b in range(1, k + 1):

            ab = GCD(a, b)

            if ab == 1:

                ans += k

                continue

            for c in range(1, k + 1):

                ans += GCD(ab, c)

    return ans
