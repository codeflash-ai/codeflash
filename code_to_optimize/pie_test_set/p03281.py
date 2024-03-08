def problem_p03281():
    import numpy as np

    a = np.zeros(201)

    for d in range(1, 201, 2):

        a[d :: 2 * d] += 1

    a = (a == 8).cumsum()

    N = int(eval(input()))

    print((a[N]))


problem_p03281()
