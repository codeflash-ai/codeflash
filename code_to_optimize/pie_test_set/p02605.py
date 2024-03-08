def problem_p02605():
    I = N = 9**9

    P, M, L, R, U, *D = "x+y,", "x-y,", [], [], []

    for t in [*open(0)][1:]:
        *z, u = t.split()
        exec(u + "+=[*map(int,z)],")

    for *A, S in (
        (D, U, "x,y,"),
        (L, R, "y,x,"),
        (U, R, P + M),
        (L, D, P + M),
        (L, U, M + P),
        (D, R, M + P),
    ):

        n = p = -I

        for x, y, r in eval("sorted((%sQ)for Q,P in enumerate(A)for x,y in P)" % S):

            if r:
                n, p = x, y

            elif x == n:
                N = min(N, y - p)

    print((N % I * 5 or "SAFE"))


problem_p02605()
