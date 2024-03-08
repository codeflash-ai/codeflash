def problem_p03132():
    L, *A = list(map(int, open(0)))

    D = [(2, 1), (1, 0), (0, 1)]

    s0 = s1 = s2 = s3 = s4 = 0

    for a in A:

        e, o = D[(a - 1) % 2 + 1 if a else 0]

        s0 += a

        s1 = s0 if s0 < s1 + e else s1 + e

        s2 = s1 if s1 < s2 + o else s2 + o

        s3 = s2 if s2 < s3 + e else s3 + e

        s4 = s3 if s3 < s4 + a else s4 + a

    print((min(s0, s1, s2, s3, s4)))


problem_p03132()
