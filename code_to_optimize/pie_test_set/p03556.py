def problem_p03556(input_data):
    import math

    n = int(eval(input_data))

    for i in range(n, 0, -1):

        if math.sqrt(i).is_integer():

            return i

            break
