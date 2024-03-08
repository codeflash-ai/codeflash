def problem_p03626():
    N = int(eval(input()))

    S1 = eval(input())

    S2 = eval(input())

    lis = []

    i = 0

    while i < N:

        if i == N - 1:

            lis.append(1)

            i += 1

        elif S1[i] == S1[i + 1]:

            lis.append(2)

            i += 2

        else:

            lis.append(1)

            i += 1

    # print(lis)

    ans = 3

    if lis[0] == 2:

        ans = 6

    for i in range(len(lis) - 1):

        if lis[i] == 2:

            if lis[i + 1] == 2:

                ans *= 3

        else:

            ans *= 2

        ans %= 10**9 + 7

    print(ans)

    # print(*ans, sep='\n')


problem_p03626()
