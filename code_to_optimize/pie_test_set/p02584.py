def problem_p02584():
    X, K, D = list(map(int, input().split()))

    if X + K * D <= 0:

        print((-X - K * D))

    elif X - K * D >= 0:

        print((X - K * D))

    else:

        div = X // D

        mod = X % D

        if (K - div) % 2 == 0:

            print(mod)

        else:

            print((abs(mod - D)))


problem_p02584()
