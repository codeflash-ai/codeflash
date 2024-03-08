def problem_p03544():
    N = int(eval(input()))

    L = [0] * 1000000

    L[0] = 2

    L[1] = 1

    for i in range(2, N + 1):

        L[i] = L[i - 2] + L[i - 1]

    print((L[N]))


problem_p03544()
