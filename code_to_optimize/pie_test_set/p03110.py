def problem_p03110():
    N = int(eval(input()))

    X = []

    U = []

    ans = 0

    for i in range(N):

        A, B = input().split()

        X.append(float(A))

        U.append(B)

    for i in range(N):

        if U[i] == "JPY":

            ans += X[i]

        elif U[i] == "BTC":

            ans += X[i] * 380000.0

    print(ans)


problem_p03110()
