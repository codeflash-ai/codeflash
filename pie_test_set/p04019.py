def problem_p04019(input_data):
    #!/usr/bin/env python3

    import sys, math, itertools, collections, bisect

    input = lambda: sys.stdin.buffer.readline().rstrip().decode("utf-8")

    inf = float("inf")
    mod = 10**9 + 7

    mans = inf
    ans = 0
    count = 0
    pro = 1

    s = eval(input_data)

    C = collections.Counter(s)

    if (C["N"] or C["S"]) and C["N"] * C["S"] == 0:

        return "No"

    elif (C["W"] or C["E"]) and C["W"] * C["E"] == 0:

        return "No"

    else:

        return "Yes"
