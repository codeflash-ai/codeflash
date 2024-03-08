def problem_p00089():
    def path(w, h, sm):

        global mx

        if w < 0 or len(inp[h]) - 1 < w:
            return 0

        sm += inp[h][w]

        if sm < mx[h][w]:
            return 0

        else:
            mx[h][w] = sm

        if h == c - 1:
            return sm

        if h < c / 2:
            return max(path(w, h + 1, sm), path(w + 1, h + 1, sm))

        else:
            return max(path(w, h + 1, sm), path(w - 1, h + 1, sm))

    c = 0

    inp = []

    while True:

        try:

            inp.append(list(map(int, input().split(","))))

            c += 1

        except:

            mx = [[0 for i in range((c + 1) / 2)] for i in range(c)]

            print(path(0, 0, 0))

            break


problem_p00089()
