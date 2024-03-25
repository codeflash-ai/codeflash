def problem_p02909(input_data):
    s = eval(input_data)

    l = ["Sunny", "Cloudy", "Rainy"]

    return l[l.index(s) - 2]
