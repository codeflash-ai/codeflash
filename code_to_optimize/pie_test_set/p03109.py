def problem_p03109():
    import datetime

    s = eval(input())

    inputDate = datetime.datetime.strptime(s, "%Y/%m/%d")

    lastDate = datetime.datetime.strptime("2019/04/30", "%Y/%m/%d")

    if inputDate > lastDate:

        print("TBD")

    else:

        print("Heisei")


problem_p03109()
