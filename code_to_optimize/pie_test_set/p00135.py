def problem_p00135():
    for _ in [0] * int(eval(input())):

        h, m = list(map(int, input().split(":")))

        s = l = 0.0

        l = m * 6

        s = 30 * (h + (m / 60))

        if l < s:
            l, s = s, l

        if l - s > 180:
            d = 360 - l + s

        else:
            d = l - s

        if d < 30:
            print("alert")

        elif d < 90:
            print("warning")

        else:
            print("safe")


problem_p00135()
