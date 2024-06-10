def problem_p03394(input_data):
    N = int(eval(input_data))

    if N == 3:

        ans = {2, 3, 25}

    elif N == 4:

        ans = {2, 3, 4, 9}

    elif N == 5:

        ans = {2, 3, 4, 6, 9}

    else:

        ans = set()

        sumAns = 0

        num = 0

        # 2の倍数もしくは3の倍数を、小さい順に追加する

        for i in range(1, 30001):

            if i % 2 == 0 or i % 3 == 0:

                ans.add(i)

                sumAns += i

                num += 1

            if num == N:
                break

        # [合計が6の倍数]になるように調整する

        if sumAns % 6 == 2:

            ans.remove(8)

            ans.add(((i + 6) // 6) * 6)  # 次の6の倍数

        elif sumAns % 6 == 3:

            ans.remove(9)

            ans.add(((i + 6) // 6) * 6)  # 次の6の倍数

        elif sumAns % 6 == 5:

            ans.remove(9)

            ans.add(((i + 2) // 6) * 6 + 4)  # 次の6k+4形式の数

    return " ".join(map(str, ans))
