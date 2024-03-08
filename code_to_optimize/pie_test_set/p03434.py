def problem_p03434():
    import numpy as np

    N = int(eval(input()))

    a = np.array(list(map(int, input().split())))

    a.sort()

    a = a[::-1]

    print((a[0::2].sum() - a[1::2].sum()))


problem_p03434()
