def problem_p03127():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    while len(A) > 1:

        A.sort()

        # print(A)

        for i in range(1, len(A)):

            # print(A[i],A[0])

            A[i] = A[i] % A[0]

        A = list([x for x in A if x != 0])

    print((A[0]))


problem_p03127()
