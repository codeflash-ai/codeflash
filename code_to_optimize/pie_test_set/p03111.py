def problem_p03111():
    import itertools

    n, a, b, c = list(map(int, input().split()))

    l = [int(eval(input())) for i in range(n)]

    ans = 10**9

    for k in itertools.product(list(range(4)), repeat=n):

        A = [[] for i in range(4)]

        for i in range(n):

            A[k[i]] += [l[i]]

        if A[1] and A[2] and A[3]:

            tmp = 10 * (n - len(A[0]) - 3)

            tmp += abs(a - sum(A[1]))

            tmp += abs(b - sum(A[2]))

            tmp += abs(c - sum(A[3]))

            ans = min(tmp, ans)

    print(ans)


problem_p03111()
