def problem_p03893(input_data):
    x = int(input_data)

    x -= 1

    ans = 6

    for i in range(x):

        ans = ans + ans + 2

    return ans
