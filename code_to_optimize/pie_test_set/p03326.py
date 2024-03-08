def problem_p03326():
    N, M = list(map(int, input().split()))

    xyz = [list(map(int, input().split())) for _ in range(N)]

    ans = [[] for _ in range(8)]

    for x, y, z in xyz:

        for i in range(8):

            ans[i].append(x * ((-1) ** i) + y * ((-1) ** (i // 2)) + z * ((-1) ** (i // 4)))

    out = []

    for i in range(8):

        ans[i].sort(reverse=True)

        out.append(sum(ans[i][:M]))

    print((max(out)))


problem_p03326()
