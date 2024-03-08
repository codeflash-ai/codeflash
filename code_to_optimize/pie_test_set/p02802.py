def problem_p02802():
    n, m = list(map(int, input().split()))

    l = [tuple(input().split()) for _ in range(m)]

    subs = [[] for _ in range(n)]

    for i in range(m):

        subs[int(l[i][0]) - 1].append(l[i][1])

    ac, p = 0, 0

    for j in range(n):

        if "AC" in subs[j]:

            ac += 1

            for k in range(len(subs[j])):

                if subs[j][k] == "WA":

                    p += 1

                else:

                    break

    print((ac, p))


problem_p02802()
