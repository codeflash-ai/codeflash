def problem_p02836(input_data):
    s = eval(input_data)

    res = 0

    for i in range(len(s) // 2):

        res += int(s[i] != s[len(s) - i - 1])

    return res
