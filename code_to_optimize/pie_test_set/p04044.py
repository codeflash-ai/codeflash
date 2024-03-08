def problem_p04044():
    N, L = list(map(int, input().split()))

    S = list(eval(input()) for i in range(N))

    str = ""

    for i in range(N - 1):

        for j in range(N - 1):

            if S[j + 1] + S[j] <= S[j] + S[j + 1]:

                S[j], S[j + 1] = S[j + 1], S[j]

    for i in S:

        str += i

    print(str)


problem_p04044()
