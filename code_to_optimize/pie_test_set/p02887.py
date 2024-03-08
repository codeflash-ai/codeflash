def problem_p02887():
    n = int(eval(input()))

    s = str(eval(input()))

    slist = []

    for i in range(n):

        slist.append(s[i])

    slime = slist[0]

    count = 1

    for i in range(1, n):

        if slime != slist[i]:

            count += 1

            slime = slist[i]

    print(count)


problem_p02887()
