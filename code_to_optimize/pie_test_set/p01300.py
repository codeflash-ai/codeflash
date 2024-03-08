def problem_p01300():
    while True:

        S = input()

        if S == "0":
            break

        m = [1] + [0 for i in range(10)]

        d = A = 0

        e = 1

        S = reversed(S)

        for c in S:

            d = (d + int(c) * e) % 11

            if int(c) != 0:
                A += m[d]

            m[d] += 1

            e *= -1

        print(A)


problem_p01300()
