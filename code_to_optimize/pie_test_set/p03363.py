def problem_p03363():
    n = int(eval(input()))

    A = [int(i) for i in input().split()]

    S = [0] * (n + 1)

    num = {}

    for i in range(n):

        S[i + 1] = S[i] + A[i]

    for i in S:

        if str(i) in list(num.keys()):

            num[str(i)] += 1

        else:

            num[str(i)] = 1

    out = [i * (i - 1) // 2 for i in list(num.values()) if i > 1]

    print((sum(out)))


problem_p03363()
