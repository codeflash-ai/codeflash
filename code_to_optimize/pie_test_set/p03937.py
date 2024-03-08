def problem_p03937():
    H, W = list(map(int, input().split()))

    A = [list(eval(input())) for _ in range(H)]

    cnt = 0

    for i in range(H):

        cnt += A[i].count("#")

    print(("Possible" if cnt == H + W - 1 else "Impossible"))


problem_p03937()
