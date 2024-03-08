def problem_p03470():
    N = int(eval(input()))

    d = set(sorted([int(eval(input())) for _ in range(N)]))

    print((len(d)))


problem_p03470()
