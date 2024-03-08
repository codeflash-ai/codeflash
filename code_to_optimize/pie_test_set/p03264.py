def problem_p03264():
    n = int(eval(input()))

    m = list(range(1, n + 1))

    even = []

    odd = []

    for x in range(0, len(m)):

        if len(m) == 0:

            print("0")

        else:

            if m[x] % 2 == 0:

                even.append(m[x])

            else:

                odd.append(m[x])

    output = [[a, b] for a in even for b in odd]

    print((len(output)))


problem_p03264()
