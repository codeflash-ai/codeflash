def problem_p03341():
    N = int(eval(input()))

    S = eval(input())

    wnums = [0]

    enums = [0]

    for i in range(0, N - 1):

        if S[i] == "W":

            wnums.append(wnums[i] + 1)

        else:

            wnums.append(wnums[i])

    j = 0

    for i in range(N - 1, 0, -1):

        if S[i] == "E":

            enums.append(enums[j] + 1)

        else:

            enums.append(enums[j])

        j += 1

    enums.reverse()

    ans = 10**9

    for i in range(N):

        num = wnums[i] + enums[i]

        if ans > num:

            ans = num

    print(ans)


problem_p03341()
