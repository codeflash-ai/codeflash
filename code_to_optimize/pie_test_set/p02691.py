def problem_p02691():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    a_index = []

    for i in range(n):

        a_index.append((i, a[i]))

    dic1 = {}

    dic2 = {}

    for i in range(n):

        one = a_index[i][0] + a_index[i][1]

        two = a_index[i][0] - a_index[i][1]

        if one not in dic1:

            dic1[one] = 1

        else:

            dic1[one] += 1

        if two not in dic2:

            dic2[two] = 1

        else:

            dic2[two] += 1

    ans = 0

    for i in list(dic1.keys()):

        if i in dic2:

            ans += dic2[i] * dic1[i]

    for i in list(dic2.keys()):

        if i in dic1:

            ans += dic1[i] * dic2[i]

            # print(i)

    print((ans // 2))


problem_p02691()
