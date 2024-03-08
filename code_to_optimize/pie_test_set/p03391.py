def problem_p03391():
    import sys

    sys.setrecursionlimit(10**7)

    def I():
        return int(sys.stdin.readline().rstrip())

    def MI():
        return list(map(int, sys.stdin.readline().rstrip().split()))

    def LI():
        return list(map(int, sys.stdin.readline().rstrip().split()))  # 空白あり

    def LI2():
        return list(map(int, sys.stdin.readline().rstrip()))  # 空白なし

    def S():
        return sys.stdin.readline().rstrip()

    def LS():
        return list(sys.stdin.readline().rstrip().split())  # 空白あり

    def LS2():
        return list(sys.stdin.readline().rstrip())  # 空白なし

    N = I()

    ans = 0

    A, B = [], []

    for i in range(N):

        a, b = MI()

        A.append(a)

        B.append(b)

    if A == B:

        print((0))

        exit()

    m = sum(B)

    for i in range(N):

        if A[i] > B[i]:

            m = min(m, B[i])

    print((sum(B) - m))


problem_p03391()
