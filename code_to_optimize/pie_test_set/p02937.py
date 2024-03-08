def problem_p02937():
    from collections import defaultdict

    from bisect import bisect_left

    s = eval(input())

    n = len(s)

    s = s + s

    t = eval(input())

    if set(t) - set(s):

        print((-1))

        exit()

    d = defaultdict(list)

    for i in range(2 * n):

        d[s[i]] += [i]

    cur = tot = 0

    for c in t:

        x = d[c][bisect_left(d[c], cur)]

        if x < n:

            cur = x + 1

        else:

            cur = x - n + 1

            tot += n

    tot += cur

    print(tot)


problem_p02937()
