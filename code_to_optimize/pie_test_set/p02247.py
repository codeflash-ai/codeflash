def problem_p02247():
    T = eval(input())

    P = eval(input())

    idx = 0

    ans = 1000

    while ans != -1:

        ans = T.find(P, idx)

        if ans == -1:

            break

        else:

            print(ans)

            if ans == 0:

                idx += 1

            else:

                idx = ans + 1


problem_p02247()
