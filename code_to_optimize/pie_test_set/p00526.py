def problem_p00526():
    n = int(input())

    il = list(map(int, input().split()))

    s = 0

    t = 0

    cnt = []

    while t < n - 1:

        if il[t] == il[t + 1]:

            cnt.append(t + 1 - s)

            s = t + 1

        t += 1

        if t == n - 1:

            cnt.append(t + 1 - s)

    ans = 0

    if len(cnt) <= 3:

        print((sum(cnt)))

    else:

        for i in range(len(cnt) - 2):

            ans = max(ans, cnt[i] + cnt[i + 1] + cnt[i + 2])

        print(ans)


problem_p00526()
