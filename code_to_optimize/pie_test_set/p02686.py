def problem_p02686():
    N = int(eval(input()))

    D, E = [], []

    t, l = 0, 0

    res = 0

    for _ in range(N):

        S = input().rstrip()

        x, y = 0, 0

        for s in S:

            if s == "(":
                x += 1

            else:
                x = max(x - 1, 0)

        for s in reversed(S):

            if s == ")":
                y += 1

            else:
                y = max(y - 1, 0)

        D.append((x, y))

    D.sort(key=lambda x: x[1])

    t = 0

    for x, y in D:

        if x - y >= 0:

            if t >= y:
                t += x - y

            else:
                print("No")
                exit()

    D.sort(key=lambda x: x[0])

    s = 0

    for x, y in D:

        if y - x >= 0:

            if s >= x:
                s += y - x

            else:
                print("No")
                exit()

    if t != s:
        print("No")

    else:
        print("Yes")


problem_p02686()
