def problem_p03381():
    import numpy as np

    N = int(eval(input()))

    X = np.array(list(map(int, input().split())))

    sort_X = np.sort(X)

    l = sort_X[int(N / 2 - 1)]

    r = sort_X[int(N / 2)]

    for i in range(N):

        if X[i] <= l:

            print(r)

        else:

            print(l)


problem_p03381()
