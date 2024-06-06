def problem_p03024(input_data):
    S = eval(input_data)

    return "YES" if 7 - len(S) >= -sum([c == "o" for c in S]) else "NO"
