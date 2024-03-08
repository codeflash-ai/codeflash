def problem_p03800():
    N = int(eval(input()))

    S = eval(input())

    dic = {
        ("S", "S", "o"): "S",
        ("S", "S", "x"): "W",
        ("S", "W", "o"): "W",
        ("S", "W", "x"): "S",
        ("W", "S", "o"): "W",
        ("W", "S", "x"): "S",
        ("W", "W", "o"): "S",
        ("W", "W", "x"): "W",
    }

    cnd = ["SS", "SW", "WS", "WW"]

    for h in cnd:

        rlt = h

        l = dic[(h[0], h[1], S[0])]

        for i in range(1, N - 1):

            rlt += dic[(rlt[i], rlt[i - 1], S[i])]

        if l == rlt[-1] and dic[(rlt[-1], rlt[-2], S[-1])] == rlt[0]:

            print(rlt)

            exit()

    print((-1))


problem_p03800()
