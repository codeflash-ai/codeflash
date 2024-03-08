def problem_p00637():
    while 1:

        n = eval(input())

        if n == 0:
            break

        p = list(map(int, input().split()))

        sp = p[0]

        ans = []

        for i in range(n - 1):

            if p[i] + 1 != p[i + 1]:

                if sp != p[i]:

                    ans.append(str(sp) + "-" + str(p[i]))

                else:

                    ans.append(str(sp))

                sp = p[i + 1]

        else:

            if sp != p[-1]:

                ans.append(str(sp) + "-" + str(p[-1]))

            else:

                ans.append(str(sp))

        print(" ".join(ans))


problem_p00637()
