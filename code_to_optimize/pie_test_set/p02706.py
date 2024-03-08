def problem_p02706():
    NM = list(map(int, input().split()))

    a = list(map(int, input().split()))

    x = 0

    for i in range(NM[1]):

        x += a[i]

    if x > NM[0]:

        print((-1))

    else:

        print((NM[0] - x))


problem_p02706()
