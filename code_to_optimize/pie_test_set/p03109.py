def problem_p03109(input_data):
    import datetime

    s = eval(input_data)

    inputDate = datetime.datetime.strptime(s, "%Y/%m/%d")

    lastDate = datetime.datetime.strptime("2019/04/30", "%Y/%m/%d")

    if inputDate > lastDate:

        return "TBD"

    else:

        return "Heisei"
