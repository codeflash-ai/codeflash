def problem_p02880():
    N = int(eval(input()))

    # N>81のときはn*nで表せないので不敵

    if N > 81:

        print("No")

    else:

        import numpy as np

        n = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        amari = N % n

        shou = N // n

        if np.any((amari == 0) * (shou <= 9)):

            print("Yes")

        else:

            print("No")


problem_p02880()
