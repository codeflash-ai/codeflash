def problem_p03714():
    from heapq import heapify, heappush, heappop

    N = eval(input())

    A = list(map(int, input().split()))

    q = A[:N]

    heapify(q)

    q_sum = sum(q)

    B = [float("-inf") for i in range(3 * N)]

    B[N - 1] = q_sum

    for i in range(N, 2 * N):

        x = A[i]

        if x > q[0]:

            tmp = heappop(q)

            q_sum -= tmp

            q_sum += x

            heappush(q, x)

        B[i] = q_sum

    q = [x * -1 for x in A[2 * N :]]

    heapify(q)

    q_sum = sum(q) * -1

    B[2 * N - 1] -= q_sum

    for i in range(2 * N - 1, N - 1, -1):

        x = A[i] * -1

        if q[0] < x:

            tmp = heappop(q)

            q_sum -= tmp * -1

            q_sum += x * -1

            heappush(q, x)

        B[i - 1] -= q_sum

    print(max(B))


problem_p03714()
