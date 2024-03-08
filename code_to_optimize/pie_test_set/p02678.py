def problem_p02678():
    n, m = list(map(int, input().split()))

    li = [[] for i in range(n)]

    for i in range(m):

        a, b = list(map(int, input().split()))

        li[a - 1].append(b - 1)

        li[b - 1].append(a - 1)

    sign = {0: 0}

    step = {0: 1}

    now = {0}

    flg = False

    while len(now) > 0:

        if flg:

            print("No")

            exit()

        flg = True

        next = set()

        for i in now:

            for j in li[i]:

                if not j in sign:

                    sign[j] = i

                    step[j] = step[i] + 1

                    next.add(j)

                    flg = False

        now.clear()

        now = next

    print("Yes")

    for i in range(n - 1):

        print((sign[i + 1] + 1))


problem_p02678()
