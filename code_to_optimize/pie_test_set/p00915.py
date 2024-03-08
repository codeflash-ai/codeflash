def problem_p00915():
    while 1:

        n, l = list(map(int, input().split()))

        if n == 0:
            break

        tube = [[] for i in range(l - 1)]

        for i in range(1, n + 1):

            d, p = input().split()

            tube[int(p) - 1].append(i if d == "R" else -i)

        t = num = 0

        while sum(len(ele) for ele in tube) != 0:

            for s in [-1, 1]:

                for i in range(l - 1)[::s]:

                    for a in tube[i]:

                        if -s * a > 0:

                            tube[i].remove(a)

                            if i == (l - 2 if s == -1 else 0):

                                num = abs(a)

                            else:

                                tube[i - s].append(a)

            for i in range(l - 1):

                if len(tube[i]) > 1:

                    tube[i] = [-a for a in tube[i]]

            t += 1

        print(t, num)


problem_p00915()
