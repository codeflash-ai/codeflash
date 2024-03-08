def problem_p02360():
    from itertools import accumulate

    N = int(eval(input()))

    xys = []

    m_x = 0

    m_y = 0

    for i in range(N):

        xy = list(map(int, input().split()))

        xys += [xy]

        x_1, y_1, x_2, y_2 = xy

        m_x = max(x_2, m_x)

        m_y = max(y_2, m_y)

    im = [[0] * (m_x + 1) for i in range(m_y + 1)]

    for xy in xys:

        x_1, y_1, x_2, y_2 = xy

        im[y_1][x_1] += 1

        im[y_1][x_2] -= 1

        im[y_2][x_1] -= 1

        im[y_2][x_2] += 1

    print((max(list(map(max, list(map(accumulate, list(zip(*list(map(accumulate, im)))))))))))


problem_p02360()
