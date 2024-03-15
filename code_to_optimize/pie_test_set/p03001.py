def problem_p03001(input_data):
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

    W, H, x, y = MI()

    if 2 * x == W and 2 * y == H:

        return (H * W / 2, 1)

    else:

        return (H * W / 2, 0)
