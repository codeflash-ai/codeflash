def problem_p00992():
    a = sorted([int(eval(input())) for _ in [0] * int(eval(input()))])

    c = b = 0

    for i in range(len(a) - 1, -1, -1):

        if a[i] < c // 4:
            continue

        else:
            b += a[i] - c // 4

        c += 1

    print((b + 1))


problem_p00992()
