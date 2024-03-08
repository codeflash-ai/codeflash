def problem_p03207():
    n = int(eval(input()))

    a = [int(eval(input())) for i in range(n)]

    print((int(sum(a) - max(a) / 2)))


problem_p03207()
