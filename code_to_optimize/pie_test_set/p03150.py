def problem_p03150():
    S = eval(input())

    def f():

        if S == "keyence":

            print("YES")

        else:

            n = len(S)

            for i in range(n):

                for j in range(i, n):

                    T = ""

                    for k in range(n):

                        if i <= k <= j:

                            continue

                        else:

                            T += S[k]

                    if T == "keyence":

                        print("YES")

                        return

            else:

                print("NO")

    f()


problem_p03150()
