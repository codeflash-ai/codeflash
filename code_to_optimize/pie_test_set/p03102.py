def problem_p03102():
    import numpy as np

    n, m, c = list(map(int, input().split()))

    cond = list(map(int, input().split()))

    char = []

    for i in range(n):

        char.append(list(map(int, input().split())))

    npcond = np.asarray(cond)

    npchar = np.asarray(char)

    npstat = npchar * npcond

    npstat1 = np.sum(npstat, axis=1) + c

    print((np.count_nonzero(npstat1 > 0)))


problem_p03102()
