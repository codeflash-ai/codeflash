def problem_p03303():
    S = eval(input())

    w = int(eval(input()))

    newS = [""] * ((len(S) - 1 + w) // w)

    for i in range(len(newS)):

        newS[i] = S[w * i]

    print(("".join(newS)))


problem_p03303()
