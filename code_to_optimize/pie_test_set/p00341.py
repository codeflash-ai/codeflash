def problem_p00341(input_data):
    e = sorted(list(map(int, input_data.split())))

    return "yes" if len(set(e[:4])) == len(set(e[4:8])) == len(set(e[8:])) == 1 else "no"
