def problem_p00076():
    A = [[]] * 1001

    x = 1.0

    y = 0.0

    for i in range(1, 1001):

        A[i] = [x, y]

        tmp = i**0.5

        x, y = x - y / tmp, y + x / tmp

    while 1:

        n = eval(input())

        if n == -1:
            break

        print(A[n][0])

        print(A[n][1])


problem_p00076()
