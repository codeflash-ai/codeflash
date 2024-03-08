def problem_p03158():
    from math import ceil

    N, Q = list(map(int, input().split()))

    A = [int(a) for a in input().split()]

    AltTake = [0 for a in range(N)]

    GetAll = [0 for a in range(N)]

    AltTake[0], AltTake[1] = A[0], A[1]

    GetAll[-1] = A[-1]

    BlockList = [0 for i in range(ceil(N / 2))]

    maxBlock = ceil(N / 2)

    for i in range(2, N):

        AltTake[i] = A[i] + AltTake[i - 2]

    for i in reversed(list(range(N - 1))):

        GetAll[i] = GetAll[i + 1] + A[i]

    for i in range(ceil(N / 2) - 1):

        BlockList[-i - 1] = (A[-i - 2] + A[-2 * i - 3]) // 2 + 1

    def BSearch(x, L):

        left, right = 0, len(L)

        while right - left > 1:

            mid = (right + left) // 2

            if L[mid] > x:

                right = mid

            else:

                left = mid

        return left

    for _ in range(Q):

        block = maxBlock - BSearch(int(eval(input())), BlockList)

        if block == maxBlock:
            print((GetAll[-block]))

        else:
            print((GetAll[-block] + AltTake[N - block * 2 - 1]))


problem_p03158()
