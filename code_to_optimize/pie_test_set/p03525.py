def problem_p03525():
    n = int(eval(input()))

    a = [0] * 13

    for i in map(int, input().split()):

        a[i] += 1

    ans = 0

    if not (any(i > 2 for i in a) or a[0] or a[12] > 1):

        s = [[0]]

        for i in range(1, 13):

            nxt = []

            if a[i] == 0:

                nxt = s

            elif a[i] == 1:

                for si in s:

                    nxt.append(si + [i])

                    nxt.append(si + [24 - i])

            else:

                for si in s:

                    nxt.append(si + [i, 24 - i])

            s = nxt

        for si in s:

            si.sort()

            tmp = float("Inf")

            for i in range(len(si)):

                k = abs(si[i] - si[i - 1])

                k = min(k, 24 - k)

                tmp = min(tmp, k)

            ans = max(ans, tmp)

    print(ans)


problem_p03525()
