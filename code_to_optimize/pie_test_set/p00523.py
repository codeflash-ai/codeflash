def problem_p00523():
    def isok(m):

        p0 = 0

        p1 = 1

        p2 = 1

        while p0 < N:

            # l0>=m ??¨???????????????p1????±???????

            if p1 <= p0:
                p1 = p0 + 1

            while d[p1] - d[p0] < m and p1 - p0 < N:

                p1 += 1

            if d[p1] - d[p0] < m:

                p0 += 1

                continue

            l0 = d[p1] - d[p0]

            # l1>=l0??¨???????????????p2????±???????

            if p2 <= p1:
                p2 = p1 + 1

            while d[p2] - d[p1] < m and p2 - p0 < N:

                p2 += 1

            if d[p2] - d[p1] < m:

                p0 += 1

                continue

            l1 = d[p2] - d[p1]

            if L - l0 - l1 >= l1:

                return True

            p0 += 1

        return False

    N = int(eval(input()))

    A = [0] * N

    L = 0

    for i in range(N):

        A[i] = int(eval(input()))

        L += A[i]

    d = [0] * (2 * N)

    for i in range(1, 2 * N):

        d[i] = d[i - 1] + A[(i - 1) % N]

    left = 1

    right = L // 3

    ans = 0

    while left <= right:

        m = (left + right) // 2

        if isok(m):

            ans = m

            left = m + 1

        else:

            right = m - 1

    print(ans)


problem_p00523()
