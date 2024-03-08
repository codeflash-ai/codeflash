def problem_p00174():
    while True:

        try:

            for i in range(3):

                line = input().strip()

                if line == "0":

                    raise Exception

                a = b = 0

                for i, s in enumerate(line):

                    if i != 0:

                        if s == "A":

                            a += 1

                        else:

                            b += 1

                if a > b:

                    a += 1

                else:

                    b += 1

                print(a, b)

        except:

            break


problem_p00174()
