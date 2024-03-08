def problem_p02620():
    from numba import njit

    from numpy import int64, zeros

    @njit("i8(i8[:],i8[:])", cache=True)
    def func(s, x):

        last = zeros(26, int64)

        score = 0

        for i, v in enumerate(x, 1):

            last[v] = i

            c = 0

            for j in range(26):

                c += s[j] * (i - last[j])

            score += s[i * 26 + v] - c

        return score

    def main():

        d, *s = list(map(int, open(0).read().split()))

        s = int64(s)

        x = s[26 * -~d : d * 27 + 26] - 1

        for d, q in s[27 * -~d :].reshape(-1, 2):

            x[d - 1] = q - 1

            print((func(s, x)))

    main()


problem_p02620()
