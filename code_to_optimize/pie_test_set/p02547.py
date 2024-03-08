def problem_p02547():
    N = int(eval(input()))

    D = [list(map(int, input().split())) for _ in range(N)]

    ans = "No"

    flag = False

    for i in range(N - 2):

        if D[i][0] == D[i][1] and D[i + 1][0] == D[i + 1][1] and D[i + 2][0] == D[i + 2][1]:
            flag = True

        if flag:

            ans = "Yes"

            break

    print(ans)


problem_p02547()
