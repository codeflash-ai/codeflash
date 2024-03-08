def problem_p04012():
    import numpy as np

    w = eval(input())

    anal = np.array([])

    for i in range(ord("a"), ord("z") + 1):

        anal = np.append(anal, w.count(chr(i)))

    if np.all(anal % 2 == 0):

        # ans = np.sum(anal)

        print("Yes")

    else:

        print("No")


problem_p04012()
