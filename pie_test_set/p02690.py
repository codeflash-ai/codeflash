def problem_p02690(input_data):
    X = int(eval(input_data))

    for A in range(-200, 200):

        for B in range(-200, 200):

            if A**5 - B**5 == X:

                return (A, B)

                break

        else:

            continue

        break
