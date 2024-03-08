def problem_p02468():
    import sys

    mod = 10**9 + 7

    def solve():

        m, n = list(map(int, input().split()))

        ans = modpow(m, n, mod)

        print(ans)

    def modpow(x, y, mod):

        res = 1

        while y:

            if y & 1:

                res = (res * x) % mod

            x = (x * x) % mod

            y >>= 1

        return res

    if __name__ == "__main__":

        solve()


problem_p02468()
