def problem_p02910(input_data):
    s = eval(input_data)

    flag = True

    for i in range(1, len(s) + 1):

        if i % 2 == 1:

            if s[i - 1] == "L":

                flag = False

        else:

            if s[i - 1] == "R":

                flag = False

    if flag:

        return "Yes"

    else:

        return "No"
