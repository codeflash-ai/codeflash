def problem_p01359():
    while 1:

        n, m = list(map(int, input().split()))

        if n == 0:
            break

        EBY = [input().split() for i in range(n)]

        EBY = [[e, int(b), int(y)] for e, b, y in EBY]

        for i in range(m):

            year = int(input())

            for j in range(n):

                if EBY[j][2] - EBY[j][1] < year <= EBY[j][2]:

                    print(EBY[j][0], year - EBY[j][2] + EBY[j][1])

                    break

            else:

                print("Unknown")


problem_p01359()
