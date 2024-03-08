def problem_p03611():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    a = sorted(a)

    li = [[a[0], 1]]

    for i in range(1, n):

        if a[i] != a[i - 1]:

            li.append([a[i], 1])

        else:

            li[len(li) - 1][1] += 1

    ans = 0

    for i in range(len(li)):

        ans_tmp = li[i][1]

        if 0 <= i - 1:

            if li[i - 1][0] + 1 == li[i][0]:

                ans_tmp = ans_tmp + li[i - 1][1]

        if i + 1 < len(li):

            if li[i + 1][0] - 1 == li[i][0]:

                ans_tmp = ans_tmp + li[i + 1][1]

        ans = max(ans, ans_tmp)

    print(ans)


problem_p03611()
