def problem_p03698():
    import copy

    s = eval(input())

    ss = list(s)

    sss = copy.copy(ss)

    n = len(ss)

    ans = "yes"

    for i in range(n):

        sss = copy.copy(ss)

        sss.remove(sss[i])

        if ss[i] in sss:

            ans = "no"

    print(ans)


problem_p03698()
