def problem_p00629():
    while 1:

        n = eval(input())

        if n == 0:
            break

        t = [list(map(int, input().split())) for i in [1] * n]

        t = sorted(sorted(sorted(t), key=lambda x: x[3])[::-1], key=lambda x: x[2])[::-1]

        u = [0 for i in range(1001)]

        s = 0

        for i in t:

            if (s < 10 and u[i[1]] < 3) or (s < 20 and u[i[1]] < 2) or (s < 26 and u[i[1]] < 1):

                print(i[0])

                u[i[1]] += 1

                s += 1


problem_p00629()
