def problem_p02878():
    N, A, B = list(map(int, input().split()))
    M = 998244353
    P = N + 1
    f = 1
    F = [1] * P
    I = [1] * P
    z = 0
    R = range

    for n in R(1, P):
        F[n] = f = f * n % M

    I[N] = i = pow(f, M - 2, M)

    for n in R(N, 1, -1):
        I[n - 1] = i = i * n % M

    for k in R(min(A + 1, N - B) if N - B - A else A + 1):
        Q = N - B - k - 1
        z = (z + (B - k) * F[B + k - 1] * I[B] * I[k] * F[Q + A - k] * I[Q] * I[A - k]) % M

    print((z if B else 1))


problem_p02878()
