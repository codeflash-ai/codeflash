def problem_p03189():
    import sys

    n, q = list(map(int, input().split()))

    MOD = 10**9 + 7

    INV2 = (MOD + 1) // 2

    lines = sys.stdin.readlines()

    aaa = list(map(int, lines[:n]))

    mat = [[0] * n for _ in [0] * n]

    for i in range(n):

        for j in range(n):

            mat[i][j] = int(aaa[i] < aaa[j])

    # print(*mat, sep='\n')

    for line in lines[n:]:

        x, y = list(map(int, line.split()))

        x -= 1

        y -= 1

        mat[x][y] = mat[y][x] = (mat[x][y] + mat[y][x]) * INV2 % MOD

        for i in range(n):

            if i == x or i == y:

                continue

            mat[x][i] = mat[y][i] = (mat[x][i] + mat[y][i]) * INV2 % MOD

            mat[i][x] = mat[i][y] = (mat[i][x] + mat[i][y]) * INV2 % MOD

        # print(x, y)

        # print(*mat, sep='\n')

    ans = sum(sum(row[:i]) for i, row in enumerate(mat)) % MOD

    ans = (ans << q) % MOD

    print(ans)


problem_p03189()
