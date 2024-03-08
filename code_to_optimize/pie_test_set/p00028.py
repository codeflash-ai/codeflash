def problem_p00028():
    d = [0 for i in range(101)]

    while True:

        try:

            n = eval(input())

            d[n] += 1

        except:

            break

    for e in [i for i, e in enumerate(d) if e == max(d)]:

        print(e)


problem_p00028()
