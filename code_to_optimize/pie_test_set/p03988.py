def problem_p03988():
    import sys

    input = sys.stdin.readline

    sys.setrecursionlimit(1000000)

    from collections import deque

    def getN():

        return int(eval(input()))

    def getList():

        return list(map(int, input().split()))

    import math

    n = getN()

    nums = getList()

    # nums.sort(reverse=True)

    mx = max(nums)

    mn = (mx + 1) // 2

    mndx = (mx % 2) + 1

    from collections import Counter

    cnt = Counter(nums)

    for i in range(mn + 1, mx + 1):

        if cnt[i] < 2:

            print("Impossible")

            sys.exit()

    if cnt[mn] != mndx:

        print("Impossible")

    else:

        print("Possible")


problem_p03988()
