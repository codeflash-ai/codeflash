def problem_p03133():
    n, m = list(map(int, input().split()))

    a = [int(input().replace(" ", ""), 2) for i in range(n)]

    mod = 998244353

    cnt = 0

    for j in range(m)[::-1]:

        for i in range(cnt, n):

            if a[i] & 1 << j:

                for k in range(n):

                    if (i != k) and (a[k] & 1 << j):

                        a[k] ^= a[i]

                a[i], a[cnt] = a[cnt], a[i]

                cnt += 1

    if cnt == 0:

        print((0))

    else:

        ans = pow(2, n + m - 2 * cnt, mod) * pow(2, cnt - 1, mod) * (pow(2, cnt, mod) - 1) % mod

        print(ans)


problem_p03133()
