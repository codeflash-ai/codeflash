def problem_p02791():
    N = int(eval(input()))

    P = list(map(int, input().split()))

    l = []

    a = P[0]

    for i in range(N):

        if P[i] <= a:

            a = P[i]

            continue

        else:

            l.append(i)

    print((N - len(l)))


problem_p02791()
