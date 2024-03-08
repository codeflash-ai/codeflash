def problem_p03401():
    N = int(eval(input()))

    alist = list(map(int, input().split()))

    a2list = [0] + alist + [0]

    # print(a2list)

    alldist = 0

    for i in range(1, N + 2):

        alldist += abs(a2list[i] - a2list[i - 1])

    # print(alldist)

    for i in range(N):

        if a2list[i + 1] - a2list[i] >= 0 and a2list[i + 2] - a2list[i + 1] >= 0:

            print(alldist)

        elif a2list[i + 1] - a2list[i] <= 0 and a2list[i + 2] - a2list[i + 1] <= 0:

            print(alldist)

        else:

            dist = (
                alldist
                - abs(a2list[i + 1] - a2list[i])
                - abs(a2list[i + 2] - a2list[i + 1])
                + abs(a2list[i + 2] - a2list[i])
            )

            print(dist)


problem_p03401()
