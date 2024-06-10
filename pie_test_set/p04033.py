def problem_p04033(input_data):
    a, b = list(map(int, input_data.split()))

    if 0 < a:

        ans = "Positive"

    elif b < 0:

        ans = "Positive" if (b - a) % 2 == 1 else "Negative"

    else:

        ans = "Zero"

    return ans
