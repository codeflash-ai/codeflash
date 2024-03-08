def problem_p00040():
    import string

    A2Z = "abcdefghijklmnopqrstuvwxyz"

    R = list(range(26))

    def decode(msg, L):

        x = ""

        for c in msg:
            x += L[A2Z.index(c)] if "a" <= c <= "z" else c

        return x

    def affine(msg, a, b):

        L = "".join([A2Z[(i * a + b) % 26] for i in R])

        s = decode(msg, L)

        return s

    def rot(msg, a):

        a = a % 26

        L = A2Z[a:] + A2Z[:a]

        s = decode(msg, L)

        return s

    def checkkey(s):

        c0 = "t"

        for i in R:

            a = 0

            x = affine(s, i, 0)

            c = A2Z.index(x[0])

            if c != 19:

                a = (19 - c) % 26

                x = rot(x, a)

            if x in ["this", "that"]:
                return i, a

        return -1, -1

    n = eval(input())

    while n:

        n -= 1

        s = input()

        for w in s.split():

            if len(w) == 4:

                a, b = checkkey(w)

                if a != -1:
                    break

        print(affine(s, a, b))


problem_p00040()
