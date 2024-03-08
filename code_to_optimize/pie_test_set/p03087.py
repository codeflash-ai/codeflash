def problem_p03087():
    N, Q = list(map(int, input().split()))

    S = list(eval(input()))

    mark = [0] * (N + 1)

    for i in range(2, N + 1):

        if S[i - 2] == "A" and S[i - 1] == "C":

            mark[i] = mark[i - 1] + 1

        else:

            mark[i] = mark[i - 1]

    for i in range(Q):

        l, r = list(map(int, input().split()))

        l = l - 1

        r = r - 1

        print((mark[r + 1] - mark[l + 1]))


problem_p03087()
