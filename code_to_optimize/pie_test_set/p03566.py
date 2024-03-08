def problem_p03566():
    n = int(eval(input()))

    t = list(map(int, input().split()))

    v = list(map(int, input().split()))

    v.append(0)

    # forward

    now = 0

    for i in range(n):

        if now + t[i] < v[i]:

            v[i] = now + t[i]

        if v[i] <= v[i + 1]:

            now = v[i]

        elif t[i] + v[i + 1] + now < 2 * v[i]:

            v[i] = (t[i] + v[i + 1] + now) / 2

            now = v[i + 1]

        else:

            now = v[i + 1]

    # back

    now = 0

    for i in range(n - 1, -1, -1):

        if now + t[i] < v[i]:

            v[i] = now + t[i]

        if v[i] <= v[i - 1]:

            now = v[i]

        elif t[i] + v[i - 1] + now < 2 * v[i]:

            v[i] = (t[i] + v[i - 1] + now) / 2

            now = v[i - 1]

        else:

            now = v[i - 1]

    ans = 0

    now = 0

    for i in range(n):

        t1 = v[i] - now

        t3 = v[i] - v[i + 1]

        t2 = t[i] - t1 - t3

        ans += now * t1 + t1 * t1 / 2

        now = v[i]  # now += t1

        ans += now * t2

        # now = now

        ans += now * t3 - t3 * t3 / 2

        now -= t3

    print(ans)


problem_p03566()
