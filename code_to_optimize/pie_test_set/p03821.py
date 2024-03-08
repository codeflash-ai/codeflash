def problem_p03821():
    n = int(eval(input()))

    a = [list(map(int, input().split())) for _ in range(n)]

    m = 0

    for i in range(n):

        s = (a[-1 - i][0] + m) % a[-1 - i][1]

        if s != 0:
            m += a[-1 - i][1] - s

    print(m)


problem_p03821()
