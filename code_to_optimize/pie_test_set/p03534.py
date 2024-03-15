def problem_p03534(input_data):
    S = eval(input_data)

    C = {"a": 0, "b": 0, "c": 0}

    for s in S:

        C[s] += 1

    if max(C.values()) - min(C.values()) <= 1:

        return "YES"

    else:

        return "NO"
