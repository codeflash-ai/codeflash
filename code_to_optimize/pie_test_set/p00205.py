def problem_p00205():
    while 1:

        try:

            h = [eval(input()) for i in [1] * 5]

            l = set(h)

            if 1 in l and 2 in l:
                w = 1

            elif 1 in l and 3 in l:
                w = 3

            else:
                w = 2

            if len(l) != 2:

                for i in h:
                    print(3)

            else:

                for i in h:
                    print(1 if i == w else 2)

        except:
            break


problem_p00205()
