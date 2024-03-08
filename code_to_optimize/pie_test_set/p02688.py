def problem_p02688():
    N, K = list(map(int, input().split()))

    con = 0

    ans = list(int() for _ in range(100000))

    pre = 0

    for _ in range(K):

        D = int(eval(input()))

        arr = list(map(int, input().split()))

        for j in range(D):

            ans[pre + j] = arr[j]

        pre += D

    fin = (N + 1) - len(set(ans))

    print(fin)


problem_p02688()
