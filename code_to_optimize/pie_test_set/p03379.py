def problem_p03379():
    import numpy as np

    N = int(eval(input()))

    X = list(map(int, input().split()))

    X_sort = sorted(X)

    small, big = X_sort[N // 2 - 1], X_sort[N // 2]

    for x in X:

        if x >= big:

            print(small)

        else:

            print(big)


problem_p03379()
