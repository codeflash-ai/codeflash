def problem_p03679(input_data):
    x, a, b = list(map(int, input_data.split()))

    if b - a <= x and b - a <= 0:

        return "delicious"

    elif b - a <= x:

        return "safe"

    else:

        return "dangerous"
