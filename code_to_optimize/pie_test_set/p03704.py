def problem_p03704():
    def func(d, cnt9, cnt0):

        if cnt9 < 1:
            return d == 0

        n = int("9" * cnt9 + "0" * cnt0)

        return sum(
            func(d + i * n, cnt9 - 2, cnt0 + 1) * (9 - abs(i) + 1 - (cnt0 < 1))
            for i in range(-9, 10)
            if abs(d + i * n) < n
        )

    D = eval(input())

    print(sum(func(D, i, 0) * (10 - i % 2 * 9) for i in range(1, 21)))


problem_p03704()
