def problem_p03386():
    a, b, k = list(map(int, input().split()))

    if b - a < k * 2:

        for j in [i for i in range(a, b + 1)]:

            print(j)

    else:

        for j in [i for i in range(a, a + k)]:

            print(j)

        for j in [i for i in range(b - k + 1, b + 1)]:

            print(j)


problem_p03386()
