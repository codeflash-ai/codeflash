def problem_p02574():
    import numpy as np

    import numba

    from numba import njit, b1, i4, i8, f8

    @njit((i8, i8[:]), cache=True)
    def main(N, A):

        Amax = np.max(A)

        lis = np.zeros(Amax + 1, np.int64)

        for i in range(N):

            lis[A[i]] += 1

        setwise = False

        for t in range(2, Amax + 1):

            cnt = np.sum(lis[t : Amax + 1 : t])

            if cnt == N:

                return "not"

            if 2 <= cnt < N:

                setwise = True

        if setwise == True:

            return "setwise"

        return "pairwise"

    N = int(eval(input()))

    A = np.array(list(map(int, input().split())))

    print((main(N, A) + " coprime"))


problem_p02574()
