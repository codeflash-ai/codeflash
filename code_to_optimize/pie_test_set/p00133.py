def problem_p00133():
    from functools import reduce

    T = lambda d: ["".join(d) for d in zip(*d[::-1])]

    D = [input() for _ in range(8)]

    for i in range(1, 4):

        print(str(90 * i))

        print("\n".join(reduce(lambda a, b: b(a), [T] * i, D)))


problem_p00133()
