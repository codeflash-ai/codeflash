def problem_p03063():
    import sys

    N, S = int(sys.stdin.readline()), sys.stdin.readline()

    b, w = 0, S.count(".")

    ans = b + w

    for si in S:

        if si == "#":

            b += 1

        if si == ".":

            w -= 1

        ans = min(ans, b + w)

    print(ans)


problem_p03063()
