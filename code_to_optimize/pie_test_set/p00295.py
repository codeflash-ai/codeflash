def problem_p00295():
    def rotate(p, comm):

        if comm == 0:

            p[0], p[1], p[2], p[27], p[28], p[29] = p[27], p[28], p[29], p[0], p[1], p[2]

            p[14], p[15] = p[15], p[14]

            p[18], p[20] = p[20], p[18]

        elif comm == 1:

            p[2], p[5], p[8], p[21], p[24], p[27] = p[21], p[24], p[27], p[2], p[5], p[8]

            p[11], p[18] = p[18], p[11]

            p[12], p[14] = p[14], p[12]

        elif comm == 2:

            p[6], p[7], p[8], p[21], p[22], p[23] = p[21], p[22], p[23], p[6], p[7], p[8]

            p[12], p[17] = p[17], p[12]

            p[9], p[11] = p[11], p[9]

        elif comm == 3:

            p[0], p[3], p[6], p[23], p[26], p[29] = p[23], p[26], p[29], p[0], p[3], p[6]

            p[9], p[20] = p[20], p[9]

            p[15], p[17] = p[17], p[15]

    def all_eq(A, left, right):

        return all(A[i] == A[left] for i in range(left, right))

    def is_correct(p):

        return (
            all_eq(p, 9, 12)
            and all_eq(p, 12, 15)
            and all_eq(p, 15, 18)
            and all_eq(p, 18, 21)
            and all_eq(p, 0, 9)
            and all_eq(p, 21, 30)
        )

    def dfs(p, cnt, f):

        ret = 9

        if cnt == 9:

            return 9

        if is_correct(p):

            return cnt

        for k in range(4):

            if k == f:

                continue

            rotate(p, k)

            ret = min(ret, dfs(p, cnt + 1, k))

            rotate(p, k)

        return ret

    n = eval(input())

    for _ in range(n):

        p = list(map(int, input().split()))

        print(dfs(p, 0, -1))


problem_p00295()
