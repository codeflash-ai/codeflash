def problem_p03999(input_data):
    from functools import reduce
    from itertools import chain, combinations

    def eval_str(string):

        s = string.split("+")

        return int(reduce(lambda x, y: int(x) + int(y), s))

    mystr = input_data

    allsubsets = lambda n: list(
        chain(*[combinations(list(range(1, n)), ni) for ni in range(n + 1)])
    )

    l = allsubsets(len(mystr))

    # for i in xrange(len(l)):

    #    l[i] = list(l[i])

    mylist = []

    # return  l

    for x in l:

        s = ""

        prev = 0

        for y in x:

            s += mystr[prev : int(y)]

            s += "+"

            prev = int(y)

        s += mystr[prev:]

        mylist.append(s)

    # return  mylist

    tot = 0

    for x in mylist:

        tot += eval_str(x)

    return tot
