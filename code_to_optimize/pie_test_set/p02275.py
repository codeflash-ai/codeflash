def problem_p02275():
    def CountingSort(A, k):

        C = [0] * (k + 1)

        n = len(A)

        for a in A:

            C[a] = C[a] + 1

        for i in range(1, k + 1):

            C[i] = C[i] + C[i - 1]

        B = [0] * n

        for j in range(n - 1, -1, -1):

            rank = C[A[j]]

            B[rank - 1] = A[j]

            C[A[j]] = C[A[j]] - 1

        return B

    n = int(eval(input()))

    A = [int(i) for i in input().split()]

    ans = CountingSort(A, 10000)

    print((" ".join([str(i) for i in ans])))


problem_p02275()
