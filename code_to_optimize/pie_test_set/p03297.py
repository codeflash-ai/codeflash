def problem_p03297():
    def solve(P):

        A, B, C, D = P

        if A < B or D < B:

            return False

        if C >= B - 1:

            return True

        g = gcd(B, D)

        return B + A % g - g <= C

    from math import gcd

    T = int(eval(input()))

    for _ in range(T):

        P = list(map(int, input().split()))

        print(("Yes" if solve(P) else "No"))


problem_p03297()
