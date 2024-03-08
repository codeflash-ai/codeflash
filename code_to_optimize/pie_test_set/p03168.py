def problem_p03168():
    # coding:utf-8

    import sys

    INF = float("inf")

    MOD = 10**9 + 7

    def LI():
        return [int(x) for x in sys.stdin.readline().split()]

    def LI_():
        return [int(x) - 1 for x in sys.stdin.readline().split()]

    def LF():
        return [float(x) for x in sys.stdin.readline().split()]

    def LS():
        return sys.stdin.readline().split()

    def II():
        return int(sys.stdin.readline())

    def SI():
        return eval(input())

    def main():

        n = II()

        P = LF()

        # dp[i]: i枚のコインが表になる確率

        dp = [1]

        # 調和級数

        # O(NlogN)

        for i in range(n):

            tmp = [0] * (i + 2)

            p = P[i]

            q = 1 - P[i]

            for j in range(i + 1):

                tmp[j] += dp[j] * q

                tmp[j + 1] += dp[j] * p

            dp = tmp

        return sum(dp[n // 2 + 1 :])

    print((main()))


problem_p03168()
