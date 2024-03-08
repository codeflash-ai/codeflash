def problem_p03027():
    MOD = 10**6 + 3

    def main():

        # preprocess

        fac = [1, 1]

        f_inv = [1, 1]

        inv = [0, 1]

        for i in range(2, MOD + 1):

            fac.append((fac[-1] * i) % MOD)

            inv.append((-inv[MOD % i] * (MOD // i)) % MOD)

            f_inv.append((f_inv[-1] * inv[-1]) % MOD)

        Q = int(eval(input()))

        for i in range(Q):

            x, d, n = list(map(int, input().split()))

            if d == 0:

                print((pow(x, n, MOD)))

                continue

            xd = (x * pow(d, MOD - 2, MOD)) % MOD

            if xd == 0 or xd + (n - 1) >= MOD:

                print((0))

                continue

            print((pow(d, n, MOD) * fac[xd + (n - 1)] * f_inv[xd - 1] % MOD))

    if __name__ == "__main__":

        main()


problem_p03027()
