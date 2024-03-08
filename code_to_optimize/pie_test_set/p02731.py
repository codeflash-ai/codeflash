def problem_p02731():
    import numpy as np

    L = np.array(int(eval(input())), dtype="float128")

    ans = (L / 3) ** 3

    print(ans)


problem_p02731()
