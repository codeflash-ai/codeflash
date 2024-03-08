def problem_p02838():
    n = int(eval(input()))

    a = [int(j) for j in input().split()]

    mod = 10**9 + 7

    m = len(bin(max(a))) - 2

    l = [[0, 0] for i in range(m)]

    for i in a:

        b = bin(i)[2:].rjust(m, "0")[::-1]

        for j in range(m):

            if b[j] == "0":

                l[j][0] += 1

            else:

                l[j][1] += 1

    ans = 0

    for i in range(m):

        ans += l[i][0] * l[i][1] * pow(2, i, mod)

        ans = ans % mod

    print(ans)


problem_p02838()
