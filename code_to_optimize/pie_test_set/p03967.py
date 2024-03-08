def problem_p03967():
    S = eval(input())

    G = P = 0

    K = 0

    for s in S:

        if P == G:

            if s == "p":

                K -= 1

            G += 1

        else:

            if s == "g":

                K += 1

            P += 1

    print(K)


problem_p03967()
