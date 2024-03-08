def problem_p03562():
    import random

    mod = 998244353

    N, X = input().split()

    N = int(N)

    A = []

    for i in range(N):

        A.append(int(eval(input()), 2))

    A.sort()

    a = A[-1]

    M = max(len(X) - 1, a.bit_length() - 1)

    base = []

    n = a.bit_length() - 1

    for i in range(M - n, -1, -1):

        base.append(a << i)

    for i in range(0, N - 1):

        a = A[i]

        for j in range(M):

            for b in base:

                a = min(a, a ^ b)

            if a == 0:

                break

            else:

                base.append(a)

                a = a << 1

    data = [0] * (M + 1)

    data2 = [0] * (M + 1)

    for b in base:

        data[b.bit_length() - 1] = b

        data2[b.bit_length() - 1] = 1

    for i in range(1, M + 1):

        data2[i] += data2[i - 1]

    data2 = [0] + data2

    # print(data)

    # print(data2)

    x = 0

    ans = 0

    n = len(X) - 1

    for i in range(len(X)):

        if X[i] == "1":

            if x >> (n - i) & 1 == 1:

                if data[n - i]:

                    ans += pow(2, data2[n - i], mod)

                    ans %= mod

            else:

                ans += pow(2, data2[n - i], mod)

                ans %= mod

                if data[n - i]:

                    x = x ^ data[n - i]

                else:

                    break

        else:

            if x >> (n - i) & 1 == 1:

                if data[n - i]:

                    x = x ^ data[n - i]

                else:

                    break

            else:

                continue

    else:

        ans += 1

        ans %= mod

    print(ans)


problem_p03562()
