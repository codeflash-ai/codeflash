def problem_p03373(input_data):
    A, B, C, X, Y = list(map(int, input_data.split()))

    m_value = float("inf")

    for ab in range(max(X, Y) + 1):

        ab2 = 2 * ab

        a = X - ab

        b = Y - ab

        tmp = C * ab2

        if a > 0:

            tmp += a * A

        if b > 0:

            tmp += b * B

        m_value = min(m_value, tmp)

    return m_value
