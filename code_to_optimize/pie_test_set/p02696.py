def problem_p02696(input_data):
    from sys import stdin

    import sys

    A, B, N = [int(x) for x in stdin.readline().rstrip().split()]

    # N, M = [int(x) for x in stdin.readline().rstrip().split()]

    # U = input_data.split()

    x = min(B - 1, N)

    ans = int((A * x) / B) - A * int(x / B)

    return ans

    # if N >= B:
