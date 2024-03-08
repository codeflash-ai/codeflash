def problem_p02315():
    N, W = list(map(int, input().split()))

    I = [[*list(map(int, input().split()))] for _ in [0] * N]

    C = [0] * -~W

    for i in range(1, N + 1):

        v, w = I[~-i]

        for j in range(W, w - 1, -1):

            t = v + C[j - w]

            if t > C[j]:
                C[j] = t

    print((C[W]))


problem_p02315()
