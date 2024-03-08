def problem_p04018():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    W = read().rstrip()

    N = len(W)

    def Z_algorithm(S):

        # 共通接頭辞の長さを返す

        N = len(S)

        arr = [0] * N

        arr[0] = N

        i, j = 1, 0

        while i < N:

            while i + j < N and S[j] == S[i + j]:

                j += 1

            arr[i] = j

            if not j:

                i += 1

                continue

            k = 1

            while i + k < N and k + arr[k] < j:

                arr[i + k] = arr[k]

                k += 1

            i += k
            j -= k

        return arr

    def is_periodic_left(W):

        Z = Z_algorithm(W)

        is_periodic = [False] * N

        for p in range(1, N // 2 + 10):

            for i in range(p, N, p):

                if Z[i] >= p:

                    is_periodic[p + i - 1] = True

                else:

                    break

        return is_periodic

    L = is_periodic_left(W)

    R = is_periodic_left(W[::-1])[::-1]

    if not L[-1]:

        answer = (1, 1)

    elif len(set(W)) == 1:

        answer = (N, 1)

    else:

        x = sum(not (x or y) for x, y in zip(L, R[1:]))

        answer = (2, x)

    print(("\n".join(map(str, answer))))


problem_p04018()
