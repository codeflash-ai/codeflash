def problem_p03457():
    import numpy as np

    n = int(eval(input()))

    input_X = [[int(i) for i in input().split()] for i in range(n)]

    X = np.array(input_X)

    pre_t = 0

    pre_x = 0

    pre_y = 0

    for i in range(n):

        x_move = abs(X[i][1] - pre_x)

        y_move = abs(X[i][2] - pre_y)

        xy_sum = x_move + y_move

        spend_time = X[i][0] - pre_t

        if xy_sum % 2 == spend_time % 2 and spend_time >= xy_sum:

            pre_t = X[i][0]

            pre_x = X[i][1]

            pre_y = X[i][2]

            if i + 1 == n:

                print("Yes")

            else:

                continue

        else:

            print("No")

            break


problem_p03457()
