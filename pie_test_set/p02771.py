def problem_p02771(input_data):
    import numpy

    A, B, C = input_data.split()

    a = A

    b = B

    c = C

    if a == b and a == c:

        return "No"

    elif a == b or b == c or a == c:

        return "Yes"

    elif a != b or b != c:

        return "No"
