def problem_p00691():
    a = 1 / 3

    while 1:

        z = int(eval(input()))

        if z == 0:
            break

        m = 0

        zz = z * z * z

        for x in range(1, int(z / pow(2, a)) + 1):

            xx = x * x * x

            y = int(pow(zz - xx, a))

            yy = y * y * y

            m = max(m, yy + xx)

        print((zz - m))


problem_p00691()
