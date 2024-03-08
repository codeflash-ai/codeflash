def problem_p03705():
    N, A, B = list(map(int, input().split()))

    if N == 1:

        print((1 if A == B else 0))

    else:

        if A > B:

            print((0))

        else:

            m = A * (N - 1) + B

            M = B * (N - 1) + A

            print((M - m + 1))


problem_p03705()
