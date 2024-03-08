def problem_p04011():
    N = int(eval(input()))

    K = int(eval(input()))

    X = int(eval(input()))

    Y = int(eval(input()))

    if N <= K:

        c = X * N

    else:

        c = X * K + Y * (N - K)

    print(c)


problem_p04011()
