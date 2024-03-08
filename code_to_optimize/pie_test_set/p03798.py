def problem_p03798():
    n = int(eval(input()))

    s = eval(input())

    x = ["SS", "SW", "WS", "WW"]

    def check0(i):

        if x[i][0] == "S":

            if s[0] == "o":

                if x[i][-1] == "S":

                    return x[i][1] == "S"

                else:

                    return x[i][1] == "W"

            else:

                if x[i][-1] == "S":

                    return x[i][1] == "W"

                else:

                    return x[i][1] == "S"

        else:

            if s[0] == "o":

                if x[i][-1] == "S":

                    return x[i][1] == "W"

                else:

                    return x[i][1] == "S"

            else:

                if x[i][-1] == "S":

                    return x[i][1] == "S"

                else:

                    return x[i][1] == "W"

    def check1(i):

        if x[i][n - 1] == "S":

            if s[n - 1] == "o":

                if x[i][n - 2] == "S":

                    return x[i][0] == "S"

                else:

                    return x[i][0] == "W"

            else:

                if x[i][n - 2] == "S":

                    return x[i][0] == "W"

                else:

                    return x[i][0] == "S"

        else:

            if s[n - 1] == "o":

                if x[i][n - 2] == "S":

                    return x[i][0] == "W"

                else:

                    return x[i][0] == "S"

            else:

                if x[i][n - 2] == "S":

                    return x[i][0] == "S"

                else:

                    return x[i][0] == "W"

    for i in range(4):

        for j in range(n - 2):

            if x[i][j + 1] == "S":

                if s[j + 1] == "o":

                    if x[i][j] == "S":

                        x[i] += "S"

                    else:

                        x[i] += "W"

                else:

                    if x[i][j] == "S":

                        x[i] += "W"

                    else:

                        x[i] += "S"

            else:

                if s[j + 1] == "o":

                    if x[i][j] == "S":

                        x[i] += "W"

                    else:

                        x[i] += "S"

                else:

                    if x[i][j] == "S":

                        x[i] += "S"

                    else:

                        x[i] += "W"

        # print(x[i])

        if check0(i) and check1(i):

            print((x[i]))

            break

    else:

        print((-1))


problem_p03798()
