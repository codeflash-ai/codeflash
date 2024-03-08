def problem_p02785():
    def solve():

        N, K = list(map(int, input().split()))

        H = list(map(int, input().split()))

        if K >= N:

            return 0

        H.sort()

        ans = sum(H[: N - K])

        return ans

    print((solve()))


problem_p02785()
