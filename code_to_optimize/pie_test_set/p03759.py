def problem_p03759(input_data):
    a, b, c = list(map(int, input_data.split()))

    if (b - a) == (c - b):

        return "YES"

    else:

        return "NO"
