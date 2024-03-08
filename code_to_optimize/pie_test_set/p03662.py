def problem_p03662():
    from collections import defaultdict

    def inpl():
        return [int(i) for i in input().split()]

    path = defaultdict(lambda: [])

    N = int(eval(input()))

    for _ in range(N - 1):

        a, b = inpl()

        path[a].append(b)

        path[b].append(a)

    inf = float("inf")

    ctr = 1

    now = {1}

    distF = [inf for _ in range(N + 1)]

    distF[1] = 0

    while True:

        for na in now.copy():

            na = now.pop()

            for nb in path[na]:

                if distF[nb] > distF[na] + 1:

                    distF[nb] = distF[na] + 1

                    now.add(nb)

                    ctr += 1

        if ctr >= N:

            break

    ctr = 1

    now = {N}

    distS = [inf for _ in range(N + 1)]

    distS[N] = 0

    while True:

        for na in now.copy():

            na = now.pop()

            for nb in path[na]:

                if distS[nb] > distS[na] + 1:

                    distS[nb] = distS[na] + 1

                    now.add(nb)

                    ctr += 1

        if ctr >= N:

            break

    D = distF[N]

    for i in range(1, N + 1):

        if distF[i] == D // 2 and distF[i] + distS[i] == D:

            St = i

        if distF[i] == D // 2 + 1 and distF[i] + distS[i] == D:

            En = i

    path[St].remove(En)

    path[En].remove(St)

    ctr = 1

    now = {1}

    distF = [inf for _ in range(N + 1)]

    distF[1] = 0

    ctr1 = 0

    while ctr1 != ctr:

        ctr1 = ctr

        for na in now.copy():

            na = now.pop()

            for nb in path[na]:

                if distF[nb] > distF[na] + 1:

                    distF[nb] = distF[na] + 1

                    now.add(nb)

                    ctr += 1

    if 2 * ctr > N:

        print("Fennec")

    else:

        print("Snuke")


problem_p03662()
