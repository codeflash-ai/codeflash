def problem_p03126():
    import numpy as np

    N, M = list(map(int, input().split()))

    menu = np.array([])

    for i in range(N):

        a = np.array(input().split())

        menu = np.append(menu, a[1:])

    menu = np.array(menu).astype(int).ravel()

    count = 0

    for i in range(1, M + 1):

        if np.sum(menu == i) == N:

            count += 1

    print(count)


problem_p03126()
