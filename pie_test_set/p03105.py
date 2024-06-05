def problem_p03105(input_data):
    A, B, C = list(map(int, input_data.split()))

    return C if B // A > C else B // A
