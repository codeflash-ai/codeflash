def problem_p03677():
    n, m = list(map(int, input().split()))

    A = list(map(int, input().split()))

    S = 0

    for i in range(n - 1):

        S += (m + A[i + 1] - A[i]) % m

    T = 2 * m

    D = [0] * (T + 10)

    data = [0] * (T + 1)

    def get(i):

        s = 0

        while i:

            s += data[i]

            i -= i & -i

        return s

    def add(i, x):

        while i <= T:

            data[i] += x

            i += i & -i

    for i in range(n - 1):

        x = A[i]
        y = A[i + 1]

        if y < x:

            y += m

        k = y - x

        if k > 1:

            D[x + 2] += 1

            D[y + 1] -= k

            D[y + 2] += k - 1

    for i in range(1, 2 * m + 5):

        D[i] += D[i - 1]

        add(i, D[i])

    res = min(S - (get(a) + get(a + m)) for a in A)

    print(res)


problem_p03677()
