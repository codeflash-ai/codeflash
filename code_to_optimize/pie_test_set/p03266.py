def problem_p03266():
    N, K = list(map(int, input().split()))

    num = [0] * K

    for i in range(1, N + 1):

        num[i % K] += 1

    res = 0

    for a in range(K):

        b = (K - a) % K

        c = (K - a) % K

        if (b + c) % K == 0:

            res += num[a] * num[b] * num[c]

    print(res)


problem_p03266()
