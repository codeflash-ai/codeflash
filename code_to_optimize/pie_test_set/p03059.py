def problem_p03059(input_data):
    a, b, t = list(map(int, input_data.split()))

    ret = 0

    for i in range(a, t + 1, a):

        ret += b

    return ret
