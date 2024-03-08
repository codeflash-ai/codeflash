def problem_p02649():
    import sys

    import numpy as np

    from numba import njit

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    @njit("(i4[::1],i4,i8[::1])", cache=True)
    def main(A, B, C):

        counts = np.zeros(1 << 18, np.int32)

        popcount = np.zeros(1 << B, np.int32)

        for i in range(B):

            popcount[1 << i : 1 << i + 1] = popcount[: 1 << i] + 1

        answer = 0

        for i in range(1 << B):

            k = popcount[i]

            t = 0

            for x in A & i:

                n = counts[x]

                counts[x] += 1

                t -= C[n]

                t += C[n + 1]

            for x in A & i:

                counts[x] = 0

            if k & 1:

                t = -t

            answer += t

        return answer

    N, K, S, T = list(map(int, readline().split()))

    A = np.array(readline().split(), np.int32)

    def convert_problem(S, T, A):

        ng = np.zeros(len(A), np.bool)

        B = np.zeros_like(A)

        n = 0

        for i in range(18):

            s, t = (S >> i) & 1, (T >> i) & 1

            if (s, t) == (0, 0):

                ng |= ((A >> i) & 1) == 1

            elif (s, t) == (1, 1):

                ng |= ((A >> i) & 1) == 0

            elif (s, t) == (1, 0):

                print((0))

                exit()

            else:

                B += ((A >> i) & 1) << n

                n += 1

        return B[~ng], n

    A, B = convert_problem(S, T, A)

    C = np.zeros((100, 100), np.int64)

    C[0, 0] = 1

    for n in range(1, 100):

        C[n, :-1] += C[n - 1, :-1]

        C[n, 1:] += C[n - 1, :-1]

    C = C[:, 1 : K + 1].sum(axis=1)

    print((main(A, B, C)))


problem_p02649()
