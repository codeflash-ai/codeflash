def problem_p03233():
    import sys, random

    input = sys.stdin.readline

    N = int(eval(input()))

    A = [-1] * N
    B = [-1] * N

    for i in range(N):

        A[i], B[i] = list(map(int, input().split()))

    ans = min(sum(A), sum(B))

    temp = [(A[i], i) for i in range(N)] + [(B[i], i) for i in range(N)]

    temp.sort()

    # print(temp)

    data = [0] * N

    for i in range(N):

        val, id = temp[i]

        data[id] += 1

    S = sum(temp[i][0] for i in range(N))

    for i in range(N):

        if data[i] == 2:

            test = S

            ans = min(ans, test)

        elif data[i] == 1:

            if temp[N - 1][1] != i:

                test = S + max(A[i], B[i]) - temp[N - 1][0]

            else:

                test = S + max(A[i], B[i]) - temp[N - 2][0]

            ans = min(ans, test)

    print(ans)


problem_p03233()
