def problem_p02939():
    S = eval(input())

    N = len(S)

    ans = 0

    pre = ""

    now = ""

    for i in range(N):

        if pre == now or now == "":

            now += S[i]

        if pre != now:

            ans += 1

            pre = now

            now = ""

    print(ans)


problem_p02939()
