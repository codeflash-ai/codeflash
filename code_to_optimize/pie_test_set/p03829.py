def problem_p03829():
    N, A, B = list(map(int, input().split()))

    X = list(map(int, input().split()))

    from collections import deque

    X = deque(X)

    cost = 0

    now = X.popleft()

    while len(X) > 0:

        if (X[0] - now) * A <= B:

            cost += (X[0] - now) * A

        else:
            cost += B

        now = X.popleft()

    print(cost)


problem_p03829()
