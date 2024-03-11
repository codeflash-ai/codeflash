def problem_p02550(input_data):
    n, x, m = list(map(int, input_data.split()))

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

        return sum(ans[:p]) + d * sum(ans[p:]) + sum(ans[p : p + e])

    else:

        return sum(ans)
