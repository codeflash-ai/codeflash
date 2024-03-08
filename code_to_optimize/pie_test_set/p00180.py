def problem_p00180():
    def f(n):

        for k, c in list(dic.items()):

            a, b = k

            if a == n:

                if not b in cost or c < cost[b]:

                    cost[b] = c

            elif b == n:

                if not a in cost or c < cost[a]:

                    cost[a] = c

        for k, c in sorted(list(cost.items()), key=lambda x: x[1]):

            if not k in res:

                res.append(k)

                ans.append(c)

                f(k)

    while True:

        n, m = list(map(int, input().split()))

        if n == m == 0:
            break

        dic = {}

        s, b, c = list(map(int, input().split()))

        dic[(s, b)] = c

        for i in range(m - 1):

            a, b, c = list(map(int, input().split()))

            dic[(a, b)] = c

        cost = {}

        res = [s]

        ans = []

        f(s)

        print(sum(ans))


problem_p00180()
