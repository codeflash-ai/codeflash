def problem_p03609():
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

    X, t = MI()

    print((max(0, X - t)))


problem_p03609()
