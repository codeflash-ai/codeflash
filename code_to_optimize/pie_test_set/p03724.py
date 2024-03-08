def problem_p03724():
    N, M = list(map(int, input().split()))

    # 木だったらなんでもOK?

    # いもす法

    edge = [0] * (N + 1)

    for _ in range(M):

        a, b = list(map(int, input().split()))

        edge[a] += 1

        edge[b] -= 1

    ans = "YES"

    for i in range(N):

        edge[i + 1] += edge[i]

        if edge[i + 1] % 2 == 1:

            ans = "NO"

            break

    print(ans)


problem_p03724()
