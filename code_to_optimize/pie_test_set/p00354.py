def problem_p00354(input_data):
    from datetime import date

    days = {i: d for i, d in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])}

    day = int(eval(input_data))

    result = date(2017, 9, day)

    index = result.weekday()

    return days[index]
