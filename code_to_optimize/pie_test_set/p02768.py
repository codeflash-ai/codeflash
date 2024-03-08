def problem_p02768():
    n, a, b = list(map(int, input().split()))

    mod = 10**9 + 7

    a = min(a, n - a)

    b = min(b, n - b)

    m = max(a, b)

    X = [0] * (m + 1)

    Y = [0] * (m + 1)

    X[0] = 1

    X[1] = n

    Y[0] = Y[1] = 1

    for i in range(2, m + 1):

        X[i] = X[i - 1] * (n - i + 1) % mod

        Y[i] = Y[i - 1] * i % mod

    ans = (
        ((pow(2, n, mod) - 1) % mod - X[a] * pow(Y[a], mod - 2, mod) % mod) % mod
        - X[b] * pow(Y[b], mod - 2, mod) % mod
    ) % mod

    print(ans)


problem_p02768()
