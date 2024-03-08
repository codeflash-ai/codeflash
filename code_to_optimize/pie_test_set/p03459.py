def problem_p03459():
    n = int(eval(input()))

    t, x, y = [], [], []

    for _ in range(n):

        t_, x_, y_ = [int(x) for x in input().split()]

        t.append(t_)

        x.append(x_)

        y.append(y_)

    t_now = 0

    p = (0, 0)

    ans = True

    for i, z in enumerate(t):

        td = t[i] - t_now

        d = abs(p[0] - x[i]) + abs(p[1] - y[i])

        tmp = td - d

        if tmp >= 0 and tmp % 2 == 0:

            p = (x[i], y[i])

            t_now = t[i]

            continue

        else:

            ans = False

            break

    if ans:

        print("Yes")

    else:

        print("No")


problem_p03459()
