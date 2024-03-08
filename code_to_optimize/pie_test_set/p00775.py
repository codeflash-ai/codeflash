def problem_p00775():
    while True:

        R, N = list(map(int, input().split()))

        if not (R | N):

            break

        geta = 20

        buildings = [0] * (geta * 2)

        for _ in range(N):

            xl, xr, h = list(map(int, input().split()))

            for i in range(xl + geta, xr + geta):

                buildings[i] = max(buildings[i], h)

        left, right = 0, 20

        for _ in range(100):

            mid = (left + right) / 2

            flag = True

            for i in range(-R + geta, R + geta):

                if i < geta:

                    y = pow(R * R - (i - geta + 1) * (i - geta + 1), 0.5)

                    flag &= buildings[i] >= y - R + mid

                else:

                    y = pow(R * R - (i - geta) * (i - geta), 0.5)

                    flag &= buildings[i] >= y - R + mid

            if flag:

                left = mid

            else:

                right = mid

        print(("{:.20f}".format(left)))


problem_p00775()
