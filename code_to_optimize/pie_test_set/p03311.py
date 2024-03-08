def problem_p03311():
    import sys

    import numpy as np

    n = int(eval(input()))

    arr = np.array(list(map(int, input().split())))

    comp = np.arange(n) + 1

    diff = arr - comp

    diff.sort()

    diff = diff - np.median(diff)

    print((int(np.abs(diff).sum())))


problem_p03311()
