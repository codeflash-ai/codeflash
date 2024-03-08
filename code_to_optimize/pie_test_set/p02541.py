def problem_p02541():
    import sys

    sys.setrecursionlimit(10**7)

    def I():
        return int(sys.stdin.readline().rstrip())

    def MI():
        return list(map(int, sys.stdin.readline().rstrip().split()))

    def LI():
        return list(map(int, sys.stdin.readline().rstrip().split()))  # 空白あり

    def LI2():
        return list(map(int, sys.stdin.readline().rstrip()))  # 空白なし

    def S():
        return sys.stdin.readline().rstrip()

    def LS():
        return list(sys.stdin.readline().rstrip().split())  # 空白あり

    def LS2():
        return list(sys.stdin.readline().rstrip())  # 空白なし

    def extgcd(a, b):

        if b:

            d, y, x = extgcd(b, a % b)

            y -= (a // b) * x

            return d, x, y

        return a, 1, 0

    # V = [(X_i, Y_i), ...]: X_i (mod Y_i)

    def remainder(V):

        x = 0
        d = 1

        for X, Y in V:

            g, a, b = extgcd(d, Y)

            x, d = (Y * b * x + d * a * X) // g, d * (Y // g)

            x %= d

        return x, d

    N = I()

    def prime_factorization(n):

        ANS = {}

        for i in range(2, int(n**0.5) + 1):

            if n % i == 0:

                r = 0  # nがiで何回割り切れるか

                while n % i == 0:

                    n //= i

                    r += 1

                ANS[i] = r

        if n != 1:

            ANS[n] = 1

        return ANS

    prime = prime_factorization(N)

    if not 2 in list(prime.keys()):

        prime[2] = 1

    else:

        prime[2] += 1

    A = list(prime.keys())

    B = list(prime.values())

    l = len(A)

    ans = 2 * N

    from itertools import product

    for a in list(product([-1, 0], repeat=l)):

        for i in range(l):

            if a[i] != 0:

                break

        else:

            continue

        V = [(a[i], A[i] ** B[i]) for i in range(l)]

        x, d = remainder(V)

        ans = min(ans, x)

    print(ans)


problem_p02541()
