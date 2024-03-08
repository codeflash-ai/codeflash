def problem_p03460():
    N, K = list(map(int, input().split()))

    import numpy as np

    m = np.zeros((3 * K, 3 * K), dtype=np.int32)

    offset = 0

    for _ in range(N):

        x, y, c = input().split()

        x, y = int(x), int(y)

        x %= 2 * K

        y %= 2 * K

        if c == "B":

            m[x, y] += 1

        else:

            m[x, y] -= 1

            offset += 1

    K2 = 2 * K

    m[K2:, :K2] = m[:K, :K2]

    np.cumsum(m[:, :K2], axis=0, out=m[:, :K2])

    m[:K2, :K2] = m[K:, :K2] - m[:K2, :K2]

    m[:K2, K2:] = m[:K2, :K]

    np.cumsum(m[:K2, :], axis=1, out=m[:K2, :])

    m[:K2, :K2] = m[:K2, K:] - m[:K2, :K2]

    m = m[:K2, :K2]

    m += np.roll(np.roll(m, K, axis=0), K, axis=1)

    print((m.max() + offset))


problem_p03460()
