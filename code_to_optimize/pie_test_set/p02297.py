def problem_p02297():
    n = int(eval(input()))

    P = []

    s = 0

    for i in range(n):
        P.append(input().split())

    P.append(P[0])

    for i in range(n):
        s += int(P[i][0]) * int(P[i + 1][1]) - int(P[i][1]) * int(P[i + 1][0])

    print((abs(s) * 0.5))


problem_p02297()
