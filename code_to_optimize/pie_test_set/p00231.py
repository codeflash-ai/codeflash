def problem_p00231():
    while True:

        n = int(eval(input()))

        if n == 0:

            break

        tlst = []

        qlst = []

        for _ in range(n):

            m, a, b = list(map(int, input().split()))

            qlst.append((m, a, b))

            tlst.append(a)

            tlst.append(b)

            tlst.append(b - 1)

        tlst = sorted(list(set(tlst)))

        tlst.sort()

        tdic = {}

        for i, t in enumerate(tlst):

            tdic[t] = i

        lent = len(tlst)

        mp = [0] * lent

        for m, a, b in qlst:

            a, b = tdic[a], tdic[b]

            mp[a] += m

            mp[b] -= m

        acc = 0

        for i in range(lent):

            acc += mp[i]

            if acc > 150:

                print("NG")

                break

        else:

            print("OK")


problem_p00231()
