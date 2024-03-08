def problem_p00019():
    from functools import reduce as R

    print((R(lambda x, y: x * y, list(range(1, int(eval(input())) + 1)))))


problem_p00019()
