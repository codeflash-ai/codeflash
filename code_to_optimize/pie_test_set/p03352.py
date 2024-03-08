def problem_p03352():
    X = int(eval(input()))

    ans = 1

    t = min(X, 100)

    for b in range(1, t):

        for p in range(1, t):

            if b**p > ans and b**p <= X:

                ans = b**p

    print(ans)


problem_p03352()
