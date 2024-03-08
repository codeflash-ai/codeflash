def problem_p03863():
    S = eval(input())

    if S[0] == S[-1]:

        print(("Second" if len(S) % 2 else "First"))

    else:

        print(("Second" if len(S) % 2 == 0 else "First"))


problem_p03863()
