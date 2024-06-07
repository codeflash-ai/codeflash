def problem_p03456(input_data):
    import math

    a, b = input_data.split()

    S = a + b

    K = int(S)

    if math.sqrt(K) % 1 == 0:

        return "Yes"

    else:

        return "No"
