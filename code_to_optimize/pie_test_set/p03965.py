def problem_p03965():
    # len(s)//2-s.count("p")

    iG = 0

    iR = 0

    for s in input().rstrip():

        if s == "g":

            if 0 < iG:

                iR += 1

                iG -= 1

            else:

                iG += 1

        else:

            if 0 < iG:

                iG -= 1

            else:

                iR -= 1

                iG += 1

    print(iR)


problem_p03965()
