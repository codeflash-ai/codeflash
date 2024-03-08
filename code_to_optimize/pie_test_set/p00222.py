def problem_p00222():
    ref = [[[10, 8], [4, 4]], [[10, 2], [4, 2]]]

    r = 10000001

    p = [1] * r

    p[0] = p[1] = 0

    for i in range(int(r**0.5)):

        if p[i]:

            p[2 * i :: i] = [0 for j in range(2 * i, r, i)]

    while 1:

        try:

            n = eval(input())

            if n == 0:
                break

            n -= 1 - n % 2

            while any(not p[n - i] for i in [0, 2, 6, 8]):

                n -= ref[p[n - 2]][p[n - 6]][p[n - 8]]

            print(n)

        except:

            pass


problem_p00222()
