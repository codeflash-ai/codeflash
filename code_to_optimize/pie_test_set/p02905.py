def problem_p02905():
    def main():

        mod = 998244353

        max_A = 10**6

        N = int(eval(input()))

        A = list(map(int, input().split()))

        is_prime = [False, True, False, False, False, True] * (max_A // 6 + 1)

        del is_prime[max_A + 1 :]

        is_prime[1:4] = False, True, True

        for i in range(5, int(max_A**0.5) + 1):

            if is_prime[i]:

                is_prime[i * i :: i] = [False] * (max_A // i - i + 1)

        primes = [p for p, is_p in enumerate(is_prime) if is_p]  # max_A 以下の素数のリスト

        g = [0] * (max_A + 1)

        for a in A:

            g[a] += a

        for p in primes:

            # 倍数集合の高速ゼータ変換みたいなやつ  O((max_A)loglog(max_A))

            # 参考: http://noshi91.hatenablog.com/entry/2018/12/27/121649

            # 大量に約数列挙みたいなことをするときはこれで高速化できる場合が多そう？（みんぷろ 2018 本戦 A - Uncommon など）

            for k in range(max_A // p, 0, -1):

                g[k] += g[k * p]

        g = [v * v % mod for v in g]

        for p in primes:

            for k, g_kp in enumerate(g[p::p], 1):

                g[k] -= g_kp

        modinv = [0, 1]

        for i in range(2, max_A + 1):

            modinv.append(mod - mod // i * modinv[mod % i] % mod)

        ans = sum((gg * minv % mod for gg, minv in zip(g, modinv)))

        ans %= mod

        ans -= sum(A)

        ans *= modinv[2]

        ans %= mod

        print(ans)

    main()


problem_p02905()
