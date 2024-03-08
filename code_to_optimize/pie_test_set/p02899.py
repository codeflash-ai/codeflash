def problem_p02899():
    import numpy as np

    N = int(input())

    As = list(map(int, input().split()))

    # for i in range(N):

    #   print(As.index(i+1) + 1, end = " ")

    for item in np.argsort(As):

        print(item + 1, end=" ")


problem_p02899()
