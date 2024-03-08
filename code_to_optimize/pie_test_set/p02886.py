def problem_p02886():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    import itertools as it

    import numpy as np

    al = list(it.combinations(a, 2))

    total = 0

    for i in al:

        total += np.prod(i)

    print(total)


problem_p02886()
