def problem_p03497():
    N, K = list(map(int, input().split()))

    A = list(map(int, input().split()))

    C = {}

    for i in range(N):

        if A[i] not in C:

            C[A[i]] = 0

        C[A[i]] += 1

    C = sorted(list(C.items()), key=lambda x: x[1], reverse=True)

    n = len(C)

    if n <= K:

        cnt = 0

    else:

        cnt = 0

        for i in range(K, n):

            cnt += C[i][1]

    print(cnt)


problem_p03497()
