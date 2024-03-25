def problem_p02640(input_data):
    x, y = list(map(int, input_data.split()))

    flg = False

    for i in range(101):

        for j in range(101):

            if (i * 2) + (j * 4) == y and (i + j) == x:

                return "Yes"

                flg = True

                break

        if flg == True:

            break

    else:

        return "No"
