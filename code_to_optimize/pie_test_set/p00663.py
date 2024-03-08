def problem_p00663():
    while 1:

        exps = input().split("|")

        if exps[0] == "#":
            break

        for i in range(len(exps)):

            v = True

            lits = exps[i][1:-1].split("&")

            for lit1 in lits:

                for lit2 in lits:

                    if (lit1 == "~" + lit2) or ("~" + lit1 == lit2):

                        v = False

            if v:

                print("yes")

                break

        else:

            print("no")


problem_p00663()
