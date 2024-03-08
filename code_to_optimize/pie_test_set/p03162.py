def problem_p03162():
    # https://atcoder.jp/contests/dp/tasks/dp_c

    import numpy as np

    N = int(eval(input()))

    Max_List = []

    N_List = [i for i in range(3)]

    for i in range(N):

        Current_List = list(map(int, input().split()))

        if i == 0:

            Max_List.append(Current_List)

        else:

            Current_Max_List = []

            for j in range(3):

                Index_List = np.array(Max_List[i - 1])

                ind = np.ones(3, dtype=bool)

                ind[j] = False

                Current_Max_List.append(max(Index_List[ind]) + Current_List[j])

            Max_List.append(Current_Max_List)

    print((max(Max_List[-1])))


problem_p03162()
