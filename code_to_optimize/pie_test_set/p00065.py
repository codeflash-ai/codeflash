def problem_p00065():
    from collections import *

    import sys

    cnt1 = Counter()

    cnt2 = Counter()

    for s in sys.stdin:

        if s == "\n":
            break

        n, d = list(map(int, s.split(",")))

        cnt1[n] += 1

    for s in sys.stdin:

        if s == "\n":
            break

        n, d = list(map(int, s.split(",")))

        cnt2[n] += 1

    for e in cnt1:

        if cnt2[e]:
            print(e, cnt1[e] + cnt2[e])


problem_p00065()
