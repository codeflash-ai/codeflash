def problem_p02922():
    A, B = list(map(int, input().split()))

    start = 1

    for i in range(20):

        if start >= B:

            print(i)

            break

        start += A - 1


problem_p02922()
