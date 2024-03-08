def problem_p02630():
    #!/usr/bin/env python3

    import sys

    from collections import Counter

    sys.setrecursionlimit(10**8)

    INF = float("inf")

    def main():

        N = int(eval(input()))

        A = list(map(int, input().split()))

        Q = int(eval(input()))

        B, C = [0] * Q, [0] * Q

        for i in range(Q):

            B[i], C[i] = list(map(int, input().split()))

        # 個数を持つ

        # Biの個数 O(1),C[i]の個数 O(1) 更新 O(1) 総和O(1)

        count = Counter(A)

        tot = sum(A)

        for i in range(Q):

            b = count[B[i]]

            c = count[C[i]]

            count[C[i]] += b

            count[B[i]] = 0

            tot = tot - B[i] * b + C[i] * b

            print(tot)

        return

    if __name__ == "__main__":

        main()


problem_p02630()
