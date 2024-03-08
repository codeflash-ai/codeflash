def problem_p01126():
    while True:

        n, m, a = list(map(int, input().split()))

        if n == 0 and m == 0 and a == 0:

            break

        hl = [list(map(int, input().split())) for _ in range(m)]

        hl.sort(key=lambda x: x[0], reverse=True)

        ans = a

        for i in range(m):

            if ans == hl[i][1]:

                ans = hl[i][2]

            elif ans == hl[i][2]:

                ans = hl[i][1]

        print(ans)


problem_p01126()
