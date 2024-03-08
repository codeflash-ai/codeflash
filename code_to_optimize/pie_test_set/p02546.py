def problem_p02546():
    S = eval(input())

    if S.endswith("s"):

        S = S + "es"

    else:

        S = S + "s"

    print(S)


problem_p02546()
