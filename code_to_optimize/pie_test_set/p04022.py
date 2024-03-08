def problem_p04022():
    import sys

    from collections import Counter

    def make_prime_checker(n):

        # nまでの自然数が素数かどうかを表すリストを返す  O(nloglogn)

        is_prime = [False, True, False, False, False, True] * (n // 6 + 1)

        del is_prime[n + 1 :]

        is_prime[1:4] = False, True, True

        for i in range(5, int(n**0.5) + 1):

            if is_prime[i]:

                is_prime[i * i :: i] = [False] * (n // i - i + 1)

        return is_prime

    def main():

        Primes = [p for p, is_p in enumerate(make_prime_checker(2200)) if is_p]

        def decomp(n):

            res1 = res2 = 1

            for p in Primes:

                cnt = 0

                while n % p == 0:

                    n //= p

                    cnt += 1

                cnt %= 3

                if cnt == 1:

                    res1 *= p

                elif cnt == 2:

                    res2 *= p

            if int(n**0.5) ** 2 == n:

                res2 *= int(n**0.5)

            else:

                res1 *= n

            return res1 * res2 * res2, res1 * res1 * res2

        N, *S = list(map(int, sys.stdin.buffer.read().split()))

        T = []

        inv_dict = {}

        for s in S:

            t, t_inv = decomp(s)

            T.append(t)

            inv_dict[t] = t_inv

        counter_T = Counter(T)

        ans = 0

        for t, t_cnt in list(counter_T.items()):

            if t == 1:

                ans += 1

                continue

            t_inv = inv_dict[t]

            t_inv_cnt = counter_T[t_inv]

            if t_cnt > t_inv_cnt or (t_cnt == t_inv_cnt and t > t_inv):

                ans += t_cnt

        print(ans)

    main()


problem_p04022()
