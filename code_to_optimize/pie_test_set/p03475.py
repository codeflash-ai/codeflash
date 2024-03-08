def problem_p03475():
    n = int(eval(input()))

    csf = [tuple(map(int, input().split())) for _ in range(n - 1)]  # csf

    for i in range(n):

        # i:起点

        t = 0  # 駅jの出発時刻, 駅n（添え字n - 1）では、この時刻がprintされるので、初期値は0とする

        for j in range(i, n - 1):

            # j:乗車する区間

            c_j, s_j, f_j = csf[j]

            # 駅jの出発時刻を求める

            if t < s_j:

                t = s_j

            elif t % f_j == 0:

                pass

                # t = t

            else:

                # 始発より後に駅に到着し、駅で電車を待つ

                t += f_j - (t % f_j)

            t += c_j  # 乗車して次の駅に到着させる

        print(t)  # 駅n（添え字n - 1）の到着時刻


problem_p03475()
