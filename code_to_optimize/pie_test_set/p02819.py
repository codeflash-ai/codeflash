def problem_p02819(input_data):
    x = int(eval(input_data))

    f = 0

    if x == 2:

        return 2

    else:

        for i in range(x, 10**6):

            for j in range(2, i // 2 + 2):

                if i % j == 0:

                    break

            else:

                f = i

                break

        return f
