def problem_p03231():
    from fractions import gcd

    N, M = list(map(int, input().split()))

    S = list(eval(input()))

    T = list(eval(input()))

    L = gcd(N, M)

    print((N * M // L if all([S[i * N // L] == T[i * M // L] for i in range(L)]) else "-1"))


problem_p03231()
