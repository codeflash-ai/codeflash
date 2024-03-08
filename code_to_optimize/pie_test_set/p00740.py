def problem_p00740():
    while True:

        n, p = list(map(int, input().split()))

        if n == p == 0:
            break

        s, c = [0] * n, p

        i = 0

        while True:

            if c:

                c -= 1

                s[i % n] += 1

                if s[i % n] == p:
                    break

            else:

                c += s[i % n]

                s[i % n] = 0

            i += 1

        print((i % n))


problem_p00740()
