def problem_p03000():
    N, X = list(map(int, input().split()))

    L = [0] + [int(x) for x in input().split()]

    D = [0] * (N + 2)

    answer = 1

    for n in range(2, N + 2):

        D[n] = D[n - 1] + L[n - 1]

        if D[n] <= X:

            answer += 1

    print(answer)


problem_p03000()
