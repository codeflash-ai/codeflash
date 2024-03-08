def problem_p00124():
    f = 0

    while 1:

        n = eval(input())

        if n == 0:
            break

        if f > 0:
            print()

        N = [""] * n

        C = [0] * n

        for i in range(n):

            s = input()

            p = s.index(" ")

            N[i] = s[:p]

            a, b, c = list(map(int, s[p + 1 :].split(" ")))

            C[i] = (i, a * 3 + c)

        C = sorted(C, key=lambda x: (-x[1], x[0]))

        for e in C:
            print("".join([N[e[0]], ",", str(e[1])]))

        f = 1


problem_p00124()
