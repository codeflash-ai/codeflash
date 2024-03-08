def problem_p02946():
    K = list(map(int, input().split()))

    import numpy as np

    tmp = []

    for i in range(K[1] - K[0] + 1, K[1] + K[0]):

        if abs(i) <= 1000000:

            print(i)


problem_p02946()
