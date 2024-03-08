def problem_p03254():
    import numpy as np

    N, x = list(map(int, input().split()))

    a = np.array(list(map(int, input().split())))

    cs = np.sort(a).cumsum()

    if cs[-1] < x:

        print((N - 1))

    elif cs[-1] == x:

        print(N)

    else:

        print((np.where(cs <= x)[0].shape[0]))


problem_p03254()
