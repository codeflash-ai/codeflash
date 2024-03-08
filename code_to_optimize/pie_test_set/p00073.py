def problem_p00073():
    while 1:

        x = eval(input())

        h = eval(input())

        if x == 0 and h == 0:
            break

        a = float(x) / 2

        s = ((a * a + h * h) ** 0.5 * 2 + x) * x

        print(s)


problem_p00073()
