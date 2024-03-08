def problem_p03752():
    import sys

    read = sys.stdin.read

    readline = sys.stdin.readline

    readlines = sys.stdin.readlines

    sys.setrecursionlimit(10**9)

    INF = 1 << 60

    def main():

        N, K, *A = list(map(int, read().split()))

        ans = INF

        for bit in range(1 << N):

            if (not (bit & 1)) or bin(bit).count("1") != K:

                continue

            total = 0

            max_height = A[0]

            for i, a in enumerate(A[1:], 1):

                if a <= max_height:

                    if bit & (1 << i):

                        total += max_height - a + 1

                        max_height += 1

                else:

                    max_height = a

            if ans > total:

                ans = total

        print(ans)

        return

    if __name__ == "__main__":

        main()


problem_p03752()
