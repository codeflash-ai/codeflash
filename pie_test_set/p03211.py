def problem_p03211(input_data):
    s = eval(input_data)

    ans = 10 * 100

    for i in range(len(s) - 2):

        if abs(753 - int(s[i : i + 3])) <= ans:

            ans = abs(753 - int(s[i : i + 3]))

    return ans
