def problem_p02647():
    import copy

    from functools import lru_cache

    import numpy as np

    from numba import njit

    # @njit

    # @njit('(i8[::1],)', cache=True)

    # def func(A):

    #     N = len(A)

    #     B = np.zeros_like(A)

    #     for i in range(N):

    #         l = max(0, i - A[i])

    #         r = min(i + A[i], N-1)

    #         B[l] += 1

    #         if r+1 < N:

    #             B[r+1] -= 1

    #     B = np.cumsum(B)

    #     return B

    def resolve():

        N, K = list(map(int, input().split()))

        A = np.array(list(map(int, input().split())))

        # N = 2 * 10**5

        # K = 2 * 10**5

        # A = np.array([0] *  N)

        @njit
        def func(A):

            # N = len(A)

            B = np.zeros_like(A)

            for i in range(N):

                l = max(0, i - A[i])

                r = min(i + A[i], N - 1)

                B[l] += 1

                if r + 1 < N:

                    B[r + 1] -= 1

            B = np.cumsum(B)

            return B

        for k in range(K):

            # print(k)

            A = func(A)

            if k >= 50:

                break

        print((" ".join(map(str, A))))

    if __name__ == "__main__":

        resolve()


problem_p02647()
