def problem_p02690():
    X = int(eval(input()))

    for A in range(-200, 200):

        for B in range(-200, 200):

            if A**5 - B**5 == X:

                print((A, B))

                break

        else:

            continue

        break


problem_p02690()
