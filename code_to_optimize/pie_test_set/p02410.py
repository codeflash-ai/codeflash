def problem_p02410():
    n, m = list(map(int, input().split()))

    # init

    A = []

    b = []

    for _ in range(n):

        A.append(list(map(int, input().split())))

    for _ in range(m):

        b.append(int(eval(input())))

    for row in range(n):

        p = 0

        for i, j in zip(A[row], b):

            p += i * j

        print(p)


problem_p02410()
