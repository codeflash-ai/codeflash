def problem_p03377(input_data):
    A, B, X = list(map(int, input_data.split()))

    return "YES" if A <= X and X <= A + B else "NO"
