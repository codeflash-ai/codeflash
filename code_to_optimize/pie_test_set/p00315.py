def problem_p00315():
    C, n = list(map(int, input().split()))

    mp = [eval(input()) for _ in range(n)]

    diff_dic = {}

    def change(x, y):

        global diff_dic

        if (x, y) in diff_dic:

            diff_dic.pop((x, y))

        else:

            diff_dic[(x, y)] = True

    for y in range(n // 2):

        for x in range(n // 2):

            base = mp[y][x]

            if mp[y][n - 1 - x] != base:

                change(n - 1 - x, y)

            if mp[n - 1 - y][x] != base:

                change(x, n - 1 - y)

            if mp[n - 1 - y][n - 1 - x] != base:

                change(n - 1 - x, n - 1 - y)

    if n % 2:

        for x in range(n // 2):

            if mp[n // 2][x] != mp[n // 2][n - 1 - x]:

                chenge(n - 1 - x, n // 2)

        for y in range(n // 2):

            if mp[y][n // 2] != mp[n - 1 - y][n // 2]:

                change(n // 2, n - 1 - y)

    ans = 0

    if not diff_dic:

        ans += 1

    for _ in range(C - 1):

        d = int(eval(input()))

        for _ in range(d):

            r, c = list(map(int, input().split()))

            r -= 1

            c -= 1

            if r < n // 2 and c < n // 2:

                change(c, n - 1 - r)

                change(n - 1 - c, r)

                change(n - 1 - c, n - 1 - r)

            elif n % 2 and r == n // 2 and c != n // 2:

                change(max(c, n - 1 - c), r)

            elif n % 2 and r != n // 2 and c == n // 2:

                change(c, max(r, n - 1 - r))

            else:

                change(c, r)

        if not diff_dic:

            ans += 1

    print(ans)


problem_p00315()
