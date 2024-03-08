def problem_p03584():
    import sys

    input = sys.stdin.readline

    def get_candidate(K):

        l = []

        b = 0

        for i in range(30)[::-1]:

            if K & (1 << i):

                l.append(b + (1 << i) - 1)

                b += 1 << i

        return l + [K]

    def get_ans(K, l):

        bit = [None] * 30

        for i in range(30):

            bit[i] = (K & (1 << i)) >> i

        ans = 0

        for i, j in l:

            for k in range(30):

                b = (i & (1 << k)) >> k

                if not ((bit[k] == 1) or (b == 0 and bit[k] == 0)):

                    break

            else:

                ans += j

        return ans

    def solve():

        N, K = list(map(int, input().split()))

        l = [tuple(map(int, input().split())) for _ in range(N)]

        ans = 0

        for k in get_candidate(K):

            ans = max(ans, get_ans(k, l))

        print(ans)

    if __name__ == "__main__":

        solve()


problem_p03584()
