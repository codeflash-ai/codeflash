def problem_p00340(input_data):
    rec = sorted(list(map(int, input_data.split())))

    return "yes" if rec[0] == rec[1] and rec[2] == rec[3] else "no"
