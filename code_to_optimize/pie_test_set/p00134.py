def problem_p00134():
    from statistics import mean

    inputCount = int(eval(input()))

    prices = [int(eval(input())) for lp in range(inputCount)]

    average = mean(prices)

    print((int(average)))


problem_p00134()
