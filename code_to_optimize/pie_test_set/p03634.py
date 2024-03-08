def problem_p03634():
    import sys

    input = sys.stdin.readline

    n = int(eval(input()))

    v = [[0] + [] * (n + 1) for i in range(n + 1)]

    for i in range(n - 1):

        a, b, c = list(map(int, input().split()))

        v[a].append([b, c])

        v[b].append([a, c])

    q, k = list(map(int, input().split()))

    l = [k]

    while l:

        now = l.pop()

        for i in v[now][1:]:

            if v[i[0]][0] == 0:

                v[i[0]][0] = i[1] + v[now][0]

                l.append(i[0])

    for i in range(q):

        x, y = list(map(int, input().split()))

        print((v[x][0] + v[y][0]))


problem_p03634()
