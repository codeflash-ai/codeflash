def problem_p03592():
    N, M, K = list(map(int, input().split()))

    for i in range(N + 1):

        for j in range(M + 1):

            t = i * M + j * N - i * j * 2

            if t == K:

                print("Yes")

                exit()

    print("No")


problem_p03592()
