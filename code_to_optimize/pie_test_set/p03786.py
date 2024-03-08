def problem_p03786():
    import numpy as np

    import sys

    n = int(eval(input()))

    arr = np.array(list(map(int, input().split())))

    arr.sort()

    arr_cs = arr.cumsum()

    ans = 0

    for i in range(n - 1):

        if arr_cs[i] * 2 < arr[i + 1]:

            ans = i + 1

    else:

        print((n - ans))


problem_p03786()
