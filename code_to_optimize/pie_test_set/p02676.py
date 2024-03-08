def problem_p02676():
    k = int(eval(input()))

    s = eval(input())

    if len(s) > k:

        print((s[:k] + "..."))

    else:

        print(s)


problem_p02676()
