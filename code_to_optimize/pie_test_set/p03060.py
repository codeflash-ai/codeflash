def problem_p03060():
    import numpy as np

    N = int(eval(input()))

    V = np.array([int(i) for i in input().split(" ")])

    C = np.array([int(i) for i in input().split(" ")])

    vc = V - C

    val = 0

    for i in range(N):

        if vc[i] > 0:

            val += vc[i]

    print(val)


problem_p03060()
