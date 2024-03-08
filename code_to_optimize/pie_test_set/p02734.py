def problem_p02734():
    K = 32

    P = 998244353

    pa = (1 << 30) - ((1 << 30) % P)

    M = []

    N, S = list(map(int, input().split()))

    m = int(("1" * 2 + "0" * 30) * (S + 1), 2)

    mm = (1 << K * (S + 1)) - 1

    A = [int(a) for a in input().split()]

    s = 0

    ans = 0

    for a in A:

        s += 1

        s += s << a * K

        s &= mm

        s -= ((s & m) >> 30) * pa

        ans += s >> S * K

    print((ans % P))


problem_p02734()
