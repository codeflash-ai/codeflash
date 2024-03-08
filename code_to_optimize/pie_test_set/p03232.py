def problem_p03232():
    def mod_inv(n, p):

        if n < 0:

            return -mod_inv(-n, p) % p

        if n > p:

            return mod_inv(n % p, p)

        def mod_inv_calc(a, b):

            if b == 0:

                return (a, 1)

            else:

                s, t = mod_inv_calc(b, a % b)

                d = a // b

                return (t, -t * d + s)

        return mod_inv_calc(p, n)[1] % p

    N = int(eval(input()))

    A = [int(a) for a in input().split()]

    p = 10**9 + 7

    inv_list = [mod_inv(i + 1, p) for i in range(N + 2)]

    fact_list = [1]

    for i in range(N):

        fact_list.append(fact_list[i] * (i + 1) % p)

    s = 0

    sum_list = [inv_list[i] * fact_list[N] % p for i in range(N + 2)]

    for i in range(N):

        s += inv_list[i]

    s *= fact_list[N]

    s %= p

    ans = 0

    for i in range(N):

        ans += A[i] * s

        ans %= p

        s += sum_list[i + 1] - sum_list[N - i - 1]

        s %= p

    print(ans)


problem_p03232()
