def problem_p02842(input_data):
    n = int(eval(input_data))
    return ([m for m in range(n + 1) if int(m * 1.08) == n] + [":("])[0]
