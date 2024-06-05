def problem_p03212(input_data):
    import itertools

    n = eval(input_data)

    c = []

    for i in range(3, len(n) + 1):

        c += list(itertools.product([3, 5, 7], repeat=i))

    m = int(n)

    ans = 0

    for i in range(len(c)):

        if 3 in c[i] and 5 in c[i] and 7 in c[i]:

            search = ""

            for g in range(len(c[i])):

                search += str(c[i][g])

            if m >= int(search):

                ans += 1

    return ans
