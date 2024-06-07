def problem_p00322(input_data):
    import itertools

    u = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    a = 0

    n = list(map(int, input_data.split()))

    for x in itertools.permutations(u):

        f = 0

        for i in range(9):

            if n[i] != -1 and n[i] != x[i]:
                f = 1

        if f:
            continue

        if x[0] + x[2] + x[5] - x[8] + (x[1] + x[4] - x[7]) * 10 + (x[3] - x[6]) * 100 == 0:
            a += 1

    return a
