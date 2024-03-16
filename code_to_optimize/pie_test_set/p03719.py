def problem_p03719(input_data):
    A, B, C = [int(x) for x in input_data.strip().split(" ")]

    if not C < A and not C > B:

        return "Yes"

    else:

        return "No"
