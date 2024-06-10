def problem_p03698(input_data):
    import copy

    s = eval(input_data)

    ss = list(s)

    sss = copy.copy(ss)

    n = len(ss)

    ans = "yes"

    for i in range(n):

        sss = copy.copy(ss)

        sss.remove(sss[i])

        if ss[i] in sss:

            ans = "no"

    return ans
