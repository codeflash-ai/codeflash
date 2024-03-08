def problem_p03649():
    from numpy import *

    N, *A = open(0)

    N, A, S, a = int(N), loadtxt(A, int), 1, 0

    while S:
        B = A // N
        S = sum(B)
        a += S
        A += S - B * -~N

    print(a)


problem_p03649()
