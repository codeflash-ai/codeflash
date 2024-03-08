def problem_p03494():
    import numpy as np

    import sys

    n = int(eval(input()))

    a = np.array(list(map(int, input().split())))

    cnt = 1

    if np.any(a % 2 == 1):

        print((0))

        sys.exit()

    while True:

        a2 = a % 2**cnt

        if np.all(a2 == 0):

            cnt += 1

            continue

        else:

            print((cnt - 1))

            break


problem_p03494()
