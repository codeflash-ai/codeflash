def problem_p03711(input_data):
    # coding: utf-8

    group = {"a": [1, 3, 5, 7, 8, 10, 12], "b": [4, 6, 9, 11], "c": [2]}

    input_lines = input_data

    input_numbers = input_lines.split()

    n = list(map(int, input_numbers))  # n[0]とn[1]に数字

    first_group = False

    second_group = False

    for i in group:

        for j in range(0, len(group[i])):

            if group[i][j] == n[0]:

                first_group = True

            if group[i][j] == n[1]:

                second_group = True

        if (
            first_group == True
            and second_group == False
            or first_group == False
            and second_group == True
        ):

            return "No"

            break

    if first_group and second_group:

        return "Yes"
