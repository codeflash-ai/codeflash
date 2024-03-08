def problem_p02679():
    import sys

    input = sys.stdin.readline

    def gcd(a, b):

        while b:

            a, b = b, a % b

        return a

    MOD = 10**9 + 7

    N = int(eval(input()))

    F = []

    zero_v = 0

    inf = 0

    zero = 0

    for _ in range(N):

        x, y = list(map(int, input().split()))

        if x == 0 and y == 0:

            zero_v += 1

        elif x == 0:

            inf += 1

        elif y == 0:

            zero += 1

        else:

            G = gcd(abs(x), abs(y))

            F.append((x // G, y // G))

    all_keys = set()

    for x, y in F:

        if x * y > 0:

            if x > 0:

                all_keys.add((x, y))

            else:

                all_keys.add((-x, -y))

        x, y = -y, x

        if x * y > 0:

            if x > 0:

                all_keys.add((x, y))

            else:

                all_keys.add((-x, -y))

    all_dic = {k: [0, 0] for k in all_keys}

    for x, y in F:

        if x * y > 0:

            if x > 0:

                all_dic[(x, y)][0] += 1

            else:

                all_dic[(-x, -y)][0] += 1

        else:

            if x > 0:

                all_dic[(-y, x)][1] += 1

            else:

                all_dic[(y, -x)][1] += 1

    ans = (pow(2, inf, MOD)) + (pow(2, zero, MOD)) - 1

    ans %= MOD

    for k in all_keys:

        P, N = all_dic[k]

        ans *= (pow(2, P, MOD)) + (pow(2, N, MOD)) - 1

        ans %= MOD

    ans += zero_v - 1

    print((ans % MOD))


problem_p02679()
