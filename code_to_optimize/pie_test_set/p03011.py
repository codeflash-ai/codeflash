def problem_p03011(input_data):
    pqr = list(map(int, input_data.split()))

    pqr.sort()

    return sum(pqr[:2])
