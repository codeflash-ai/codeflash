def problem_p00136():
    h = [float(eval(input())) for _ in range(int(eval(input())))]

    d = {"1": ":", "2": ":", "3": ":", "4": ":", "5": ":", "6": ":"}

    for v in h:

        if v < 165.0:

            d["1"] += "*"

        elif v >= 165.0 and v < 170.0:

            d["2"] += "*"

        elif v >= 170.0 and v < 175.0:

            d["3"] += "*"

        elif v >= 175.0 and v < 180.0:

            d["4"] += "*"

        elif v >= 180.0 and v < 185.0:

            d["5"] += "*"

        else:

            d["6"] += "*"

    for k, v in sorted(list(d.items()), key=lambda x: int(x[0])):

        print((k + v))


problem_p00136()
