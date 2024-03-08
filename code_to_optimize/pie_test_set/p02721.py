def problem_p02721():
    N, K, C = list(map(int, input().split()))

    S = eval(input())

    latest = [None] * K

    i = len(S) + C

    for j in range(K - 1, -1, -1):

        i = S.rindex("o", 0, i - C)

        latest[j] = i

    if i <= C or "o" not in S[: i - C]:

        i = -C - 1

        for j in latest:

            i = S.index("o", i + C + 1)

            if i == j:

                print((i + 1))


problem_p02721()
