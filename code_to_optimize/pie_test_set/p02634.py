def problem_p02634(input_data):
    a, b, c, d = list(map(int, input_data.split()))

    m = 998244353

    D = [[0] * (d + 1) for _ in range(c + 1)]

    D[a][b] = 1

    for i in range(a, c + 1):

        for j in range(b, d + 1):

            D[i][j] += (D[i][j - 1] * i + D[i - 1][j] * j - D[i - 1][j - 1] * (i - 1) * (j - 1)) % m

    return D[c][d]
