def problem_p01101():
    while 1:

        n, m = list(map(int, input().split()))

        if n == m == 0:

            break

        (*A,) = list(map(int, input().split()))

        s = 0

        for i in range(n):

            for j in range(i):

                if s < A[i] + A[j] <= m:

                    s = A[i] + A[j]

        print((s or "NONE"))


problem_p01101()
