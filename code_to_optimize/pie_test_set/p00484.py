def problem_p00484():
    def solve():

        n, k = list(map(int, input().split()))

        group_num = 10

        book_map = [[] for i in range(group_num)]

        acc_map = [[0] for i in range(group_num)]

        for i in range(n):

            c, g = list(map(int, input().split()))

            book_map[g - 1].append(c)

        for i in range(group_num):

            bmi = book_map[i]

            bmi.sort(reverse=True)

            ami = acc_map[i]

            acc = 0

            for j in range(len(bmi)):

                acc += bmi[j] + j * 2

                ami.append(acc)

        dp = [[0] * (k + 1) for i in range(group_num + 1)]

        for y in range(1, k + 1):

            for x in range(1, group_num + 1):

                accs = acc_map[x - 1]

                dp_pre = dp[x - 1]

                dp[x][y] = max([dp_pre[y - z] + accs[z] for z in range(min(y + 1, len(accs)))])

        print((dp[group_num][k]))

    solve()


problem_p00484()
