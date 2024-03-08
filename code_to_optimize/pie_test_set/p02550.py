def problem_p02550():
    n, x, m = list(map(int, input().split()))

    ans = []

    flag = False

    for i in range(n):

        if x in ans:

            v = x

            flag = True

            break

        ans.append(x)

        x = x**2 % m

    if flag:

        p = ans.index(v)

        l = len(ans) - p

        d, e = divmod(n - p, l)

        print((sum(ans[:p]) + d * sum(ans[p:]) + sum(ans[p : p + e])))

    else:

        print((sum(ans)))


problem_p02550()
