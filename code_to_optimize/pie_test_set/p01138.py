def problem_p01138():
    import re

    while True:

        n = int(input())

        if n == 0:

            break

        l = [0] * (24 * 60 * 60 + 1)

        for _ in range(n):

            h1, m1, s1, h2, m2, s2 = list(map(int, re.split(":| ", input())))

            l[h1 * 60 * 60 + m1 * 60 + s1] += 1

            l[h2 * 60 * 60 + m2 * 60 + s2] -= 1

        r = 0

        for i in range(24 * 60 * 60):

            l[i + 1] += l[i]

            r = max(r, l[i + 1])

        print(r)


problem_p01138()
