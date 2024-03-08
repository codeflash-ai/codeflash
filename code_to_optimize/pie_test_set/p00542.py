def problem_p00542():
    a, b, c, d, e, f = [int(eval(input())) for _ in range(6)]

    print((sum([a, b, c, d, max(e, f)]) - min([a, b, c, d])))


problem_p00542()
