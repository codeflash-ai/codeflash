def problem_p02583():
    from bisect import bisect_left, bisect_right

    n = int(eval(input()))

    l = list(map(int, input().split()))

    l.sort()

    ans = 0

    for i in range(n):

        for j in range(i + 1, n):

            if l[i] == l[j]:

                continue

            for k in range(j + 1, n):

                if l[i] == l[k] or l[j] == l[k]:

                    continue

                if l[i] + l[j] > l[k]:

                    ans += 1

    print(ans)


problem_p02583()
