def problem_p03274():
    import sys

    input = sys.stdin.readline

    def main():

        N, K = list(map(int, input().split()))

        X = list(map(int, input().split()))

        INF = float("inf")

        M = [INF] * N

        P = [INF] * N

        i_m, i_p = 0, 0

        for x in X:

            if x == 0:

                K -= 1

            elif x > 0:

                P[i_p] = x

                i_p += 1

            else:

                M[i_m] = -x

                i_m += 1

        if K <= 0:

            print((0))

            exit()

        M.sort()

        P.sort()

        ans = min(M[K - 1], P[K - 1])

        for n_M in range(1, K + 1):

            dist = 2 * M[n_M - 1] + P[K - n_M - 1]

            if ans > dist:

                ans = dist

        for n_P in range(1, K + 1):

            dist = 2 * P[n_P - 1] + M[K - n_P - 1]

            if ans > dist:

                ans = dist

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03274()
