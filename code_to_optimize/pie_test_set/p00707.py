def problem_p00707():
    # AOJ 1126: The Secret Number

    # Python3 2018.7.16 bal4u

    while True:

        W, H = list(map(int, input().split()))

        if W == 0:
            break

        d = [list(eval(input())) for r in range(H)]

        a = [[0 for c in range(W)] for r in range(H)]

        ans = 0

        for r in range(H):

            for c in range(W):

                if not d[r][c].isdigit():
                    continue

                a[r][c] = k = int(d[r][c])

                if c > 0 and d[r][c - 1].isdigit():

                    a[r][c] = a[r][c - 1] * 10 + k

                if r > 0 and d[r - 1][c].isdigit():

                    t = a[r - 1][c] * 10 + k

                    a[r][c] = max(a[r][c], t)

                ans = max(ans, a[r][c])

        print(ans)


problem_p00707()
