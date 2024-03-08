def problem_p01086():
    while True:

        n = int(eval(input()))

        if n == 0:

            break

        else:

            A = [5, 7, 5, 7, 7]

            ans = 0

            w = [len(eval(input())) for i in range(n)]

            for i in range(n):

                k = 0

                s = 0

                for j in range(i, n):

                    s += w[j]

                    if s == A[k]:

                        s = 0

                        k += 1

                        if k == 5:

                            ans = i + 1

                            break

                        elif s > A[k]:

                            break

                if ans != 0:

                    break

            print(ans)


problem_p01086()
