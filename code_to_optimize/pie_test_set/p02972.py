def problem_p02972():
    N = int(eval(input()))

    As = list(map(int, input().split()))

    isBalls = [0] * (N + 1)

    for k in reversed(list(range(1, N + 1))):

        num = sum([isBalls[k * i] for i in range(1, N // k + 1)])

        if num % 2 != As[k - 1]:

            isBalls[k] = 1

    M = sum(isBalls)

    anss = [i for i in range(1, N + 1) if isBalls[i] > 0]

    print(M)

    print((" ".join(map(str, anss))))


problem_p02972()
