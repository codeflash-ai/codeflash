def problem_p02571():
    S = eval(input())

    T = eval(input())

    cnt = len(S) - len(T) + 1

    res = []

    for i in range(cnt):

        res2 = 0

        for j in range(len(T)):

            if S[i + j] != T[j]:

                res2 += 1

        res.append(res2)

    print((min(res)))


problem_p02571()
