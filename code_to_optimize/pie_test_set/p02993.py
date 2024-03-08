def problem_p02993():
    #!/usr/bin/env python3

    import sys, math

    input = lambda: sys.stdin.buffer.readline().rstrip().decode("utf-8")

    sys.setrecursionlimit(10**8)

    inf = float("inf")

    ans = count = 0

    S = eval(input())

    for i in range(3):

        if S[i] == S[i + 1]:

            print("Bad")

            exit()

    print("Good")


problem_p02993()
