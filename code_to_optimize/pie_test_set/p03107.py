def problem_p03107(input_data):
    S = [int(_) for _ in eval(input_data)]

    length = len(S)

    temp = [S[0]]

    for x in S[1:]:

        if len(temp) == 0:

            temp.append(x)

        elif temp[-1] == x:

            temp.append(x)

        else:

            temp.pop(-1)

    return length - len(temp)
