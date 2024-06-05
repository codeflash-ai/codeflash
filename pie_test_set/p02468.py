def problem_p02468(input_data):
    import sys

    mod = 10**9 + 7

    def solve():

        m, n = list(map(int, input_data.split()))

        ans = modpow(m, n, mod)

        return ans

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
