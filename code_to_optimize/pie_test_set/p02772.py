def problem_p02772():
    import sys

    import numpy as np

    n = int(eval(input()))

    input = sys.stdin.readline

    a = np.array(list(map(int, input().split())))

    for i in a:

        if i % 2 == 0:

            if i % 3 != 0 and i % 5 != 0:

                print("DENIED")

                exit()

    print("APPROVED")


problem_p02772()
