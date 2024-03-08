def problem_p03476():
    # ABC084D - 2017-like Number

    from bisect import bisect_left as bsl, bisect_right as bsr

    def get_primes(n: int) -> list:

        # return a list of primes <= n

        sieve = [1] * n

        for i in range(3, int(n**0.5) + 1, 2):

            if sieve[i]:

                sieve[i * i :: 2 * i] = [0] * ((n - i * i - 1) // (2 * i) + 1)

        return [2] + [i for i in range(3, n, 2) if sieve[i]]

    def main():

        Q, *LR = map(int, open(0).read().split())

        P = get_primes(10**5 + 1)

        S = set(P)

        sel = [p for p in P if (p + 1) // 2 in S]  # 2017-like primes (selected P)

        # the number of 2017-like primes b/w l, r -> bisect[r] - bisect[l]

        ans = [bsr(sel, r) - bsl(sel, l) for l, r in zip(*[iter(LR)] * 2)]

        print(*ans, sep="\n")

    if __name__ == "__main__":

        main()


problem_p03476()
