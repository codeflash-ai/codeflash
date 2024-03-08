def problem_p00718():
    a = {"m": 1000, "c": 100, "x": 10, "i": 1}

    for _ in range(int(eval(input()))):

        b, s, t = eval(input()), 0, 1

        for x in b:

            if x == " ":
                continue

            if x in a:
                s += a[x] * t
                t = 1

            else:
                t = int(x)

        ans = ""

        for k in ["m", "c", "x", "i"]:

            c, s = divmod(s, a[k])

            if c:
                ans += ["", str(c)][c != 1] + k

        print(ans)


problem_p00718()
