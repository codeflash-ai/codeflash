def problem_p03273():
    H, W = list(map(int, input().split()))

    HW_list = [list(eval(input())) for i in range(H)]

    import copy

    tmp = copy.deepcopy(HW_list)

    h_index = 0

    for i, HW in enumerate(HW_list):

        is_all_shiro = True

        for hw in HW:

            if hw == "#":

                is_all_shiro = False

                h_index += 1

                break

        if is_all_shiro:

            tmp.pop(h_index)

    w_index = 0

    ans = copy.deepcopy(tmp)

    for w in range(W):

        is_all_shiro = True

        for t in tmp:

            if t[w] == "#":

                is_all_shiro = False

                w_index += 1

                break

        if is_all_shiro:

            for a in ans:

                a.pop(w_index)

    for A in ans:

        print(("".join(A)))


problem_p03273()
