def problem_p02625(input_data):
    n, m = list(map(int, input_data.split()))

    a, M, d = 1, 10**9 + 7, [1]

    for i in range(1, n + 1):

        d.append(((m - n + i - 1) * d[i - 1] + (i - 1) * d[i - 2]) % M)

        a = a * (m - i + 1) % M

    return a * d[-1] % M
