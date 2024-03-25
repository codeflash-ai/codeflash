def problem_p02987(input_data):
    S = eval(input_data)

    judge = True

    for i in range(4):

        counter = int(0)

        for j in range(4):

            if S[i] == S[j]:

                counter += 1

        if counter != 2:

            judge = False

    if judge:

        return "Yes"

    else:

        return "No"
