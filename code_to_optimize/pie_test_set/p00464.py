def problem_p00464():
    while True:

        h, w, n = list(map(int, input().split()))

        if not h:
            break

        sss = [[1 if s == "1" else 0 for s in input().split()] for i in range(h)]

        dp = [[0 for i in range(w + 1)] for j in range(h + 1)]

        dp[0][0] = n - 1

        for x in range(h):

            for y in range(w):

                a = dp[x][y]

                if sss[x][y]:

                    if a % 2:

                        dp[x + 1][y] += a // 2

                        dp[x][y + 1] += a // 2 + 1

                    else:

                        dp[x + 1][y] += a // 2

                        dp[x][y + 1] += a // 2

                else:

                    if a % 2:

                        dp[x + 1][y] += a // 2 + 1

                        dp[x][y + 1] += a // 2

                    else:

                        dp[x + 1][y] += a // 2

                        dp[x][y + 1] += a // 2

                sss[x][y] = (sss[x][y] + dp[x][y]) % 2

        #  print(sss)

        #  print(dp)

        x = y = 0

        while x < h and y < w:

            if sss[x][y]:

                y += 1

            else:

                x += 1

        print((x + 1, y + 1))


problem_p00464()
