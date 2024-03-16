def problem_p03729(input_data):
    A, B, C = input_data.split()

    if A[-1] == B[0] and B[-1] == C[0]:

        return "YES"

    else:

        return "NO"
