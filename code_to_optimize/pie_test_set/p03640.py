def problem_p03640():
    import sys

    import numpy as np

    h, w = [int(x) for x in sys.stdin.readline().split()]

    n = int(eval(input()))

    a = [int(x) for x in sys.stdin.readline().split()]

    s = h * w

    l = []

    for i, x in enumerate(a):

        for j in range(x):

            l.append(i + 1)

    l = np.array(l).reshape((h, w))

    for i, x in enumerate(l):

        if i % 2 == 0:

            print((" ".join(map(str, x))))

        else:

            print((" ".join(map(str, reversed(x)))))


problem_p03640()
