def problem_p03939():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    N, D, X = list(map(int, read().split()))

    answer = 0

    for n in range(N, 0, -1):

        mean = D + X * (n + n - 1) / 2

        answer += mean

        D = ((n + n + 2) * D + 5 * X) / (2 * n)

        X = (n + 2) * X / n

    print(answer)


problem_p03939()
