def problem_p02796():
    N = int(eval(input()))

    LR = [[int(i) for i in input().split()] for _ in range(N)]

    # 区間を右端の小さい順にソート

    LR = sorted([(x - l, x + l) for x, l in LR], key=lambda x: x[1])

    ans = 0

    # 現在選んでいる区間の内、最も右にある区間の右端

    cur_R = -float("inf")

    for i in range(N):

        # 区間が被るとき

        if cur_R > LR[i][0]:

            continue

        ans += 1

        cur_R = LR[i][1]

    print(ans)


problem_p02796()
