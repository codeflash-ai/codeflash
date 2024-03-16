def problem_p02663(input_data):
    HM = list(map(int, input_data.split()))

    hours = HM[2] - HM[0]

    minutes = HM[3] - HM[1]

    K = HM[4]

    totalTimes = hours * 60 + minutes

    return totalTimes - K
