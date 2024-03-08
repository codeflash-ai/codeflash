def problem_p03370():
    N, X = list(map(int, input().split()))

    mn = [0] * N

    for i in range(N):

        mn[i] = int(eval(input()))

    ans = N

    x = X - sum(mn)

    while x >= min(mn):

        if x // min(mn) == 0:

            mn[mn.index(min(mn))] = 1000000

        else:

            if x >= min(mn):

                x -= min(mn)

                ans += 1

    print(ans)


problem_p03370()
