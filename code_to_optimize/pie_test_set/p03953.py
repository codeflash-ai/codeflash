def problem_p03953():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    import numpy as np

    """
    
    ・対象のうさぎの期待値だけ分かれば、次時点での期待値が分かる
    
    ・期待値の差分Yを持つ。操作は、Y[i-1]とY[i]の交換
    
    """

    N = int(readline())

    X = np.zeros(N + 1, np.int64)

    X[1:] = readline().split()

    M, K = list(map(int, readline().split()))

    A = list(map(int, read().split()))

    Y = np.diff(X)

    # newY[i] = Y[P[i]]

    P = list(range(N))

    for a in A:

        P[a], P[a - 1] = P[a - 1], P[a]

    P = np.array(P)

    def mult_perm(P, Q):

        return P[Q]

    def power_perm(P, N):

        if N == 0:

            return np.arange(len(P))

        Q = power_perm(P, N // 2)

        Q = mult_perm(Q, Q)

        return mult_perm(P, Q) if N & 1 else Q

    Q = power_perm(P, K)

    newY = Y[Q]

    newX = np.cumsum(newY)

    print(("\n".join(newX.astype(str))))


problem_p03953()
