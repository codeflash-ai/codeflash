def problem_p00069():
    def main():

        while True:

            n = int(eval(input()))

            if not n:

                break

            start = int(eval(input())) - 1  # 開始の位置

            goal = int(eval(input())) - 1  # あたりの位置

            d = int(eval(input()))

            nums = [[i for i in range(n)]]  # 各段での数字の並び

            bars = []  # 横棒リスト

            for i in range(d):

                s = eval(input())

                bars.append(s)

                new = nums[-1][:]

                for j in range(n - 1):

                    if s[j] == "1":  # 横棒があれば入れ替え

                        new[j], new[j + 1] = new[j + 1], new[j]

                nums.append(new)

            to_goal = nums[-1][goal]  # あたりにたどり着く初期位置

            if to_goal == start:  # 元からあたりにたどり着く場合

                print((0))

            else:  # 各段について、to_goalとスタートが隣り合うかチェック

                for i, status in enumerate(nums[1:]):

                    to_goal_ind = status.index(to_goal)  # to_goalの位置

                    start_ind = status.index(start)  # startの位置

                    ind1, ind2 = min(to_goal_ind, start_ind), max(to_goal_ind, start_ind)

                    if ind2 - ind1 == 1:  # 隣あっていた場合

                        if (
                            (bars[i][ind1] == "0")
                            and (ind1 == 0 or bars[i][ind1 - 1] == "0")
                            and (ind2 == n - 1 or bars[i][ind2] == "0")
                        ):  # 隣り合う横棒がなければ(入れ替え可能ならば)出力

                            print((i + 1, ind1 + 1))

                            break

                else:

                    print((1))

    main()


problem_p00069()
