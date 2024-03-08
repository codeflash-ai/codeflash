def problem_p04019():
    #!/usr/bin/env python3

    import sys, math, itertools, collections, bisect

    input = lambda: sys.stdin.buffer.readline().rstrip().decode("utf-8")

    inf = float("inf")
    mod = 10**9 + 7

    mans = inf
    ans = 0
    count = 0
    pro = 1

    s = eval(input())

    C = collections.Counter(s)

    if (C["N"] or C["S"]) and C["N"] * C["S"] == 0:

        print("No")

    elif (C["W"] or C["E"]) and C["W"] * C["E"] == 0:

        print("No")

    else:

        print("Yes")


problem_p04019()
