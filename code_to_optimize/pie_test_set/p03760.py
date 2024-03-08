def problem_p03760():
    O = eval(input())

    E = eval(input())

    P = ""

    for i in range(min(len(O), len(E))):

        if O[i] == "":

            break

        else:

            P = P + O[i]

        if E[i] == "":

            break

        else:

            P = P + E[i]

    if len(O) - len(E) == 1:

        P = P + O[-1]

    elif len(E) - len(O) == 1:

        P = P + E[-1]

    else:

        P = P

    print(P)


problem_p03760()
