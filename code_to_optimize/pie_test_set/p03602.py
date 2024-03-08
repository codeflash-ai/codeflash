def problem_p03602():
    N = int(eval(input()))

    A = [list(map(int, input().split())) for i in range(N)]

    INF = 10**9 + 1

    for i in range(N):

        A[i][i] = INF

    ans = 0

    for i in range(N):

        Ai = A[i]

        for j in range(i):

            d = min(list(map(sum, list(zip(Ai, A[j])))))

            if A[i][j] > d:

                print("-1")

                exit()

            elif A[i][j] < d:

                ans += A[i][j]

    print(ans)


problem_p03602()
