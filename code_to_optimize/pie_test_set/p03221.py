def problem_p03221():
    import numpy as np

    N, M = list(map(int, input().split()))

    P = []

    Y = []

    for _ in range(M):

        p, y = list(map(int, input().split()))

        P.append(p)

        Y.append(y)

    P = np.array(P)

    Y = np.array(Y)

    PY = np.c_[P, Y]

    sort_ind = np.argsort(Y)

    PY_sort = PY[sort_ind]

    count = [0] * (int(1e5) + 1)

    id_list = []

    for p, y in PY_sort:

        count[p] += 1

        id_list.append(str(p).zfill(6) + str(count[p]).zfill(6))

    id_list = np.array(id_list)

    id_list = id_list[np.argsort(sort_ind)]

    for city_id in id_list:

        print(city_id)


problem_p03221()
