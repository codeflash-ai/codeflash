def problem_p03177():
    import sys

    sys.setrecursionlimit(1 << 25)

    readline = sys.stdin.buffer.readline

    read = sys.stdin.readline  # 文字列読み込む時はこっち

    import numpy as np

    from functools import partial

    array = partial(np.array, dtype=np.int64)

    zeros = partial(np.zeros, dtype=np.int64)

    full = partial(np.full, dtype=np.int64)

    def ints():
        return np.fromstring(readline(), sep=" ", dtype=np.int64)

    def read_matrix(H, W) -> np.ndarray:
        """return np.ndarray shape=(H,W) matrix"""

        lines = []

        for _ in range(H):

            lines.append(read())

        lines = " ".join(lines)  # byte同士の結合ができないのでreadlineでなくreadで

        return np.fromstring(lines, sep=" ", dtype=np.int64).reshape(H, W)

    MOD = 10**9 + 7

    INF = 2**31  # 2147483648 > 10**9

    # default import

    N, K = ints()

    A = read_matrix(N, N)

    # from numba import njit

    # @njit('i8[:,:](i8[:,:],i8[:,:])', cache=True)

    def matmul_mod(X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        MOD = 10**9 + 7

        # return X @ Y

        ret = np.empty((X.shape[0], Y.shape[1]), dtype=np.int64)

        for i in range(X.shape[0]):  # Xの行

            for j in range(Y.shape[1]):  # Yの列

                tmp = (X[i, :] * Y[:, j]) % MOD

                ret[i, j] = tmp.sum() % MOD

        return ret

    A_double = np.asanyarray(A, dtype=np.int64)

    ans = np.eye(N, dtype=np.int64)

    for j in range(int(K).bit_length()):

        if (K >> j) & 1:

            ans = matmul_mod(ans, A_double)

        A_double = matmul_mod(A_double, A_double)

    print((ans.sum() % MOD))


problem_p03177()
