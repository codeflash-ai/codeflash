def problem_p02916():
    import numpy as np

    N = int(eval(input()))

    A = list(map(int, input().split()))

    A = list(np.array(A) - 1)

    B = list(map(int, input().split()))

    C = list(map(int, input().split()))

    sum_ = 0

    diff = list(np.array(A[1:]) - np.array(A[:-1]))

    diff.insert(0, -1)

    for i in range(N):

        sum_ += B[A[i]]

        if diff[i] == 1:

            sum_ += C[A[i - 1]]

    print(sum_)


problem_p02916()
