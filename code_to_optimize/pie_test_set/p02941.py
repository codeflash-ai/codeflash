def problem_p02941():
    import heapq

    N = int(eval(input()))

    A = list(map(int, input().split()))

    B = list(map(int, input().split()))

    if any(a > b for a, b in zip(A, B)):

        print((-1))

        exit()

    def pb(i):

        return i - 1 if i - 1 >= 0 else N - 1

    def nx(i):

        return i + 1 if i + 1 < N else 0

    hq = []

    for i, (a, b) in enumerate(zip(A, B)):

        if a <= b - B[pb(i)] - B[nx(i)]:

            hq.append((-b, i))

    heapq.heapify(hq)

    ans = 0

    while hq:

        _, i = heapq.heappop(hq)

        if B[i] == A[i]:
            continue

        l = B[pb(i)]

        r = B[nx(i)]

        if B[i] - l - r < A[i]:
            continue

        k = (B[i] - A[i]) // (l + r)

        ans += k

        B[i] -= k * (l + r)

        ll = B[pb(pb(i))]

        rr = B[nx(nx(i))]

        if A[nx(i)] <= r - B[i] - rr:

            heapq.heappush(hq, (-r, nx(i)))

        if A[pb(i)] <= l - B[i] - ll:

            heapq.heappush(hq, (-l, pb(i)))

    if all(a == b for a, b in zip(A, B)):

        print(ans)

    else:

        print((-1))


problem_p02941()
