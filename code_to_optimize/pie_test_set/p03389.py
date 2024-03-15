def problem_p03389(input_data):
    def noofrupees(x, y, z):

        if x == y and x == z:

            return 0

        elif x == y:

            if x < z:

                return z - x

            else:

                if x % 2 == 0 and z % 2 == 0:

                    return int((x - z) / 2)

                elif x % 2 != 0 and z % 2 != 0:

                    return int((x - z) / 2)

                else:

                    z = z - 1

                    return int((x - z) / 2) + 1

        elif x == z:

            if x < y:

                return y - x

            else:

                if x % 2 == 0 and y % 2 == 0:

                    return int((x - y) / 2)

                elif x % 2 != 0 and y % 2 != 0:

                    return int((x - y) / 2)

                else:

                    y = y - 1

                    return int((x - y) / 2) + 1

        elif y == z:

            if y < x:

                return x - y

            else:

                if y % 2 == 0 and x % 2 == 0:

                    return int((y - x) / 2)

                elif y % 2 != 0 and x % 2 != 0:

                    return int((y - x) / 2)

                else:

                    x = x - 1

                    return int((y - x) / 2) + 1

    # t= int(input_data)

    t = 1

    for v in range(t):

        x, y, z = list(map(int, input_data.split()))

        if x != y and x != z and y != z:

            list1 = [x, y, z]

            list1.sort()

            s = list1[2] - list1[1]

            list1[0] += s

            list1[1] += s

            return s + noofrupees(list1[0], list1[1], list1[2])

        else:

            return noofrupees(x, y, z)
