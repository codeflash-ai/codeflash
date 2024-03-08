def problem_p02693():
    import sys

    input = sys.stdin.readline

    from collections import *

    K = int(eval(input()))

    A, B = list(map(int, input().split()))

    for i in range(A, B + 1):

        if i % K == 0:

            print("OK")

            exit()

    print("NG")


problem_p02693()
