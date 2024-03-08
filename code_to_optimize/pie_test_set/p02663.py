def problem_p02663():
    HM = list(map(int, input().split()))

    hours = HM[2] - HM[0]

    minutes = HM[3] - HM[1]

    K = HM[4]

    totalTimes = hours * 60 + minutes

    print((totalTimes - K))


problem_p02663()
