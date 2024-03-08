def problem_p02696():
    from sys import stdin

    import sys

    A, B, N = [int(x) for x in stdin.readline().rstrip().split()]

    # N, M = [int(x) for x in stdin.readline().rstrip().split()]

    # U = input().split()

    x = min(B - 1, N)

    ans = int((A * x) / B) - A * int(x / B)

    print(ans)

    # if N >= B:


problem_p02696()
