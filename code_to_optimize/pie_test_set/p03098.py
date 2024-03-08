def problem_p03098():
    N, K = list(map(int, input().split()))
    p = [int(i) - 1 for i in input().split()]
    q = [int(i) - 1 for i in input().split()]

    def I(s):

        r = [0 for i in range(N)]

        for i in range(N):
            r[s[i]] = i

        return r

    def T(s, t):

        r = [0 for i in range(N)]

        for i in range(N):
            r[i] = s[t[i]]

        return r

    m = [[0 for i in range(N)] for i in range(6)]
    A = T(T(T(q, I(p)), I(q)), p)
    X = (K - 1) // 6

    for i in range(N):
        m[0][i] = p[i]
        m[1][i] = q[i]

    for i in range(2, 6):
        m[i] = T(m[i - 1], I(m[i - 2]))

    def E(s, k):

        if k == 0:

            return [i for i in range(N)]

        elif k == 1:

            return s

        elif k % 2 == 0:

            return E(T(s, s), k // 2)

        else:

            return T(E(T(s, s), k // 2), s)

    t = E(A, X)
    a = T(T(t, m[(K - 1) % 6]), I(t))
    print((" ".join([str(i + 1) for i in a])))


problem_p03098()
