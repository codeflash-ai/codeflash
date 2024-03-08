def problem_p02934():
    import numpy as np

    n = int(eval(input()))

    a = list(map(int, input().split()))

    lista = 0

    for i in range(n):

        lista = lista + 1 / a[i]

    print((1 / lista))


problem_p02934()
