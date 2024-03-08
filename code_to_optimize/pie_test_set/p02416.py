def problem_p02416():
    res = []

    while True:

        s = eval(input())

        if s == "0":

            break

        res.append(sum([int(x) for x in s]))

    for e in res:

        print(e)


problem_p02416()
