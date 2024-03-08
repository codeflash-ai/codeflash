def problem_p03946():
    from collections import defaultdict

    N, T = list(map(int, input().split()))

    A = [int(x) for x in input().split()]

    B = [0] * N

    for i in range(N - 1):

        a = A[N - i - 1]

        B[i + 1] = max(B[i], a)

    B = B[::-1]

    c = defaultdict(int)

    for i in range(N - 1):

        c[B[i] - A[i]] += 1

    print((c[max(c)]))


problem_p03946()
