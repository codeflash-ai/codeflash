def problem_p00354():
    from datetime import date

    days = {i: d for i, d in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])}

    day = int(eval(input()))

    result = date(2017, 9, day)

    index = result.weekday()

    print((days[index]))


problem_p00354()
