def problem_p00445():
    while 1:

        try:

            a = input()

            cjoi = cioi = 0

            for i in range(len(a) - 2):

                if a[i : i + 3] == "JOI":

                    cjoi += 1

                if a[i : i + 3] == "IOI":

                    cioi += 1

            print(cjoi)

            print(cioi)

        except:

            break


problem_p00445()
