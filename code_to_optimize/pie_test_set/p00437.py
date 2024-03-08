def problem_p00437():
    while 1:

        a, b, c = list(map(int, input().split()))

        if not a:

            break

        n = eval(input())

        I = []
        J = []
        K = []

        res = [2] * (a + b + c)

        for p in range(n):

            i, j, k, r = list(map(int, input().split()))

            i -= 1
            j -= 1
            k -= 1

            if r:

                res[i] = res[j] = res[k] = 1

            else:

                I.append(i)
                J.append(j)
                K.append(k)

        for p in range(len(I)):

            i, j, k = I[p], J[p], K[p]

            if res[i] == res[j] == res[k] == 1:

                continue

            if res[i] == res[j] == 1:

                res[k] = 0

            elif res[j] == res[k] == 1:

                res[i] = 0

            elif res[k] == res[i] == 1:

                res[j] = 0

        for i in range(a + b + c):

            print(res[i])


problem_p00437()
