def problem_p02685(input_data):
    # coding: utf-8

    def solve(*args: str) -> str:

        n, m, k = list(map(int, args[0].split()))

        mod = 998244353

        if m == 1 and n - 1 == k:

            return str(1)

        ncr = 1

        p = m * pow(m - 1, n - 1, mod) % mod

        ret = p

        inv = pow(m - 1, mod - 2, mod)

        for i in range(1, k + 1):

            ncr = (ncr * (n - i) * pow(i, mod - 2, mod)) % mod

            p = (p * inv) % mod

            ret += p * ncr % mod

        return str(ret % mod)

    if __name__ == "__main__":

        return solve(*(open(0).input_data.splitlines()))
