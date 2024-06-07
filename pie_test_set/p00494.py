def problem_p00494(input_data):
    s = eval(input_data)

    length = len(s)

    ind = 0

    ans = 0

    while ind < length:

        j_num = 0

        o_num = 0

        i_num = 0

        while ind < length and s[ind] != "J":

            ind += 1

        while ind < length and s[ind] == "J":

            j_num += 1

            ind += 1

        while ind < length and s[ind] == "O":

            o_num += 1

            ind += 1

        while ind < length and s[ind] == "I":

            i_num += 1

            ind += 1

        if o_num <= i_num and o_num <= j_num:

            ans = max(ans, o_num)

    return ans
