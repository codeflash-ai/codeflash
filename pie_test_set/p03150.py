def problem_p03150(input_data):
    S = eval(input_data)

    def f():

        if S == "keyence":

            return "YES"

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

                        return "YES"

                        return

            else:

                return "NO"

    f()
