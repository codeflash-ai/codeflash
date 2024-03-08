def problem_p00039():
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    try:

        while True:

            x = []

            line = input()

            tmp = roman[line[0]]

            x.append(tmp)

            for c in line[1:]:

                tmp = roman[c]

                if x[-1] < tmp:
                    x[-1] = -x[-1]

                x.append(tmp)

            print(sum(x))

    except:

        pass


problem_p00039()
