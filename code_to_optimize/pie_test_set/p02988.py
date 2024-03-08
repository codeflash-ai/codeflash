def problem_p02988():
    import copy

    import numpy as np

    S = eval(input())

    N = int(S)

    S = eval(input())

    P = list(map(int, S.split()))

    check = []

    count = 0

    for i in range(0, N - 2, 1):

        check = P[i : i + 3]

        check = np.array(check)

        check = check.argsort()

        if check[1] == 1:

            count = count + 1

    print(count)


problem_p02988()
