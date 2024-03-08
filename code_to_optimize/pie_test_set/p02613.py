def problem_p02613():
    from collections import Counter

    N = int(eval(input()))

    S = [eval(input()) for i in range(N)]

    a = [0] * 4

    for i in range(N):

        if S[i] == "AC":

            a[0] += 1

        elif S[i] == "WA":

            a[1] += 1

        elif S[i] == "TLE":

            a[2] += 1

        else:

            a[3] += 1

    print(("AC", "x", a[0]))

    print(("WA", "x", a[1]))

    print(("TLE", "x", a[2]))

    print(("RE", "x", a[3]))


problem_p02613()
