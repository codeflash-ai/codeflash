def problem_p00827():
    def judge(x, y):

        global xx, yy, xy, axby

        if min(x, y) >= 0 and x + y <= xy:

            if x + y < xy or (x + y == xy and a * x + b * y < axby):

                xy = x + y

                axby = a * x + b * y

                xx, yy = x, y

    while 1:

        a, b, d = list(map(int, input().split()))

        if a == 0:
            break

        xy = axby = 10**10

        for y in range(50001):

            if (d + b * y) % a == 0:
                judge((d + b * y) / a, y)

            if (b * y - d) % a == 0:
                judge((b * y - d) / a, y)

            if (d - b * y) % a == 0:
                judge((d - b * y) / a, y)

        print(xx, yy)


problem_p00827()
