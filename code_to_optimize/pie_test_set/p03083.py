def problem_p03083():
    import sys

    B, W = list(map(int, sys.stdin.readline().split()))

    MOD = 10**9 + 7

    L = B + W

    fact = [1] * (L + 1)

    rfact = [1] * (L + 1)

    for i in range(L):

        fact[i + 1] = r = fact[i] * (i + 1) % MOD

        rfact[i + 1] = pow(r, MOD - 2, MOD)

    rev2 = pow(2, MOD - 2, MOD)

    p = q = 0

    base = 1

    ans = ["%d\n" % rev2]

    for i in range(1, B + W):

        base = rev2 * base % MOD

        if i - B >= 0:

            p = (p + (fact[i - 1] * rfact[B - 1] * rfact[i - B] % MOD) * base % MOD) % MOD

        if i - W >= 0:

            q = (q + (fact[i - 1] * rfact[W - 1] * rfact[i - W] % MOD) * base % MOD) % MOD

        ans.append("%d\n" % ((1 - p + q) * rev2 % MOD))

    sys.stdout.writelines(ans)


problem_p03083()
