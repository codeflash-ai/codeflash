def problem_p02793():
    from fractions import gcd

    MOD = 1000000000 + 7

    N = int(eval(input()))

    A = list(map(int, input().split()))

    l = 1

    ans = 0

    for a in A:

        g = gcd(a, l)

        l = (l // g) * a

        ans = (ans * (a // g)) % MOD

        ans = (ans + (l // a)) % MOD

    print((int(ans)))


problem_p02793()
