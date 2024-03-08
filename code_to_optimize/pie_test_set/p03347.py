def problem_p03347():
    N = int(eval(input()))

    A = [int(eval(input())) for i in range(N)]

    if A[0]:

        print((-1))

        exit()

    ans = 0

    for a, b in zip(A, A[1:]):

        if b - a > 1:

            print((-1))

            exit()

        if b == 0:
            continue

        if b - a == 1:

            ans += 1

        else:

            ans += b

    print(ans)


problem_p03347()
