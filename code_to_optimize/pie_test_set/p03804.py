def problem_p03804():
    n, m = list(map(int, input().split()))

    N = [list(eval(input())) for _ in range(n)]

    M = [list(eval(input())) for _ in range(m)]

    for i in range(n - m + 1):

        for j in range(n - m + 1):

            count = 0

            for k in range(m):

                for l in range(m):

                    if N[i + k][j + l] == M[k][l]:

                        count += 1

            if count == m**2:

                print("Yes")

                exit()

    print("No")


problem_p03804()
