def problem_p02684():
    N, K = list(map(int, input().split()))

    A = list([int(x) - 1 for x in input().split()])

    o = [0] * 63

    def bi(x):

        i = 0

        while x != 0:

            o[i] = x % 2

            x //= 2

            i += 1

    bi(K)

    n = 0

    for i in range(63):

        if o[i]:

            n = A[n]

        A = [A[A[x]] for x in range(N)]

    print((n + 1))


problem_p02684()
