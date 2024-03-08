def problem_p00593():
    case = 1

    while True:

        N = int(input())

        if N == 0:
            break

        mat = [[0 for i in range(N)] for j in range(N)]

        n = 1

        for i in range(N):  # upper half

            for j in range(i, -1, -1):

                if i % 2 == 0:

                    mat[j][i - j] = n

                    n += 1

                elif i % 2 == 1:

                    mat[i - j][j] = n

                    n += 1

        for i in range(N, 2 * N - 1):  # lower half

            for j in range(N - 1, i - N, -1):

                if i % 2 == 0:

                    mat[j][i - j] = n

                    n += 1

                elif i % 2 == 1:

                    mat[i - j][j] = n

                    n += 1

        print("Case %d:" % (case))
        case += 1

        for s in mat:

            print("".join(map(str, [str(st).rjust(3) for st in s])))


problem_p00593()
