def problem_p02418():
    p = eval(input())

    s = eval(input())

    ret = "Yes"

    try:
        (p + p).index(s)

    except:
        ret = "No"

    print(ret)


problem_p02418()
