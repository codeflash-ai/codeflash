def problem_p03576():
    inf = float("inf")

    N, K = list(map(int, input().split()))

    ans = inf

    X = []

    Y = []

    XY = []

    for _ in range(N):

        x, y = list(map(int, input().split()))

        X.append(x)

        Y.append(y)

        XY.append((x, y))

    X = sorted(X)

    Y = sorted(Y)

    for i in range(N):

        for j in range(i + 1, N):

            for k in range(N):

                for l in range(k + 1, N):

                    cnt = 0

                    for x, y in XY:

                        if X[i] <= x <= X[j] and Y[k] <= y <= Y[l]:

                            cnt += 1

                    if cnt < K:
                        continue

                    S = (Y[l] - Y[k]) * (X[j] - X[i])

                    if S < ans:

                        ans = S

    print(ans)


problem_p03576()
